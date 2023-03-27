/* ---------------------------------------------------------------------
 *
 * This program solves the linear elasticity problem on a rectangular
 * domain. It is based on step-8, but also uses step 18 and 44.
 *
 * https://www.dealii.org/current/doxygen/deal.II/step_8.html
 * https://www.dealii.org/current/doxygen/deal.II/step_18.html
 * https://www.dealii.org/current/doxygen/deal.II/step_44.html
 *
 * The problem is to find the displacement field u(x) that
 * satisfies the boundary conditions and the following equation:
 *
 *   div(sigma(u)) = 0
 *
 * where sigma(u) is the stress tensor, given by
 *
 *   sigma(u) = lambda * (div(u) I) + mu * (grad(u) + grad(u)^T)
 *
 * and lambda and mu are material constants. The boundary conditions are:
 *
 *   u = 0 on the boundary on left,
 *   u = 0.01*L on the boundary on right
 *
 * The domain is a rectangle with dimensions 2 x 2. The material constants
 * are lambda = 40 and mu = 40.
 *
 * The program uses linear finite elements and CG solver.
 *
 * ---------------------------------------------------------------------
 *
 * Capabilities:
 *
 * Multiple elements
 * Displacement field is written in vtu format to visualize in paraview.
 * Stress at the quadrature points is extracted and written in a file.
 * DOFs are written in gnuplot format.
 *
 * Future work:
 *
 * The program can be extended to write the stress to vtu format.
 * Input file can be added to read the material constants and other properties.
 *
 *
 * ---------------------------------------------------------------------
 *
 * Author: Erdem Caliskan, 2023
 * ME 660, Spring 2023
 */

// Matrix incldues
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// Solver includes
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

// Grid includes
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
// DoF includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// FE includes
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_eulerian.h>

// Numerics includes
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

// C++ includes
#include <fstream>
#include <iostream>

namespace Project
{
using namespace dealii;

// Auxiliary function to compute the stress tensor from the displacement (step-18)

// Get the 4x4x4x4 stiffness matrix for the given material constants (NOT IN USE)
// C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
/* template <int dim> SymmetricTensor<4, dim> get_stiffness(const double lambda, const double mu)
{
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) + ((i == l) && (j == k) ? mu : 0.0) +
                                       ((i == j) && (k == l) ? lambda : 0.0));
    return tmp;
} */

// Get the strain tensor from the displacement gradient of the solution
// e_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
template <int dim>
inline SymmetricTensor<2, dim> get_sym_grad(const FEValues<dim> &fe_values, const unsigned int shape_func,
                                            const unsigned int q_point)
{
    SymmetricTensor<2, dim> tmp;

    for (unsigned int i = 0; i < dim; ++i)
        tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i + 1; j < dim; ++j)
            tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                         fe_values.shape_grad_component(shape_func, q_point, j)[i])/2;

    return tmp;
}

// Get the strain tensor from the displacement gradient
// e_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
template <int dim> inline SymmetricTensor<2, dim> get_sym_grad(const std::vector<Tensor<1, dim>> &grad)
{
    Assert(grad.size() == dim, ExcInternalError());

    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
        strain[i][i] = grad[i][i];

    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i + 1; j < dim; ++j)
            strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
}

template <int dim> class StrainPostprocessor : public DataPostprocessorTensor<dim>
{
  public:
    StrainPostprocessor()
        : DataPostprocessorTensor<dim>("strain", update_values | update_gradients | update_quadrature_points)
    {
    }

    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                       std::vector<Vector<double>> &computed_quantities) const override
    {
        AssertDimension(input_data.solution_gradients.size(), computed_quantities.size());

        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
            AssertDimension(computed_quantities[p].size(), (Tensor<2, dim>::n_independent_components));
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                    computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] =
                        (input_data.solution_gradients[p][d][e] + input_data.solution_gradients[p][e][d]) / 2;
        }
    }
};

template <int dim> class StressPostprocessor : public DataPostprocessorTensor<dim>
{
  public:
    StressPostprocessor()
        : DataPostprocessorTensor<dim>("stress", update_values | update_gradients | update_quadrature_points)
    {
    }

    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                       std::vector<Vector<double>> &computed_quantities) const override
    {
        AssertDimension(input_data.solution_gradients.size(), computed_quantities.size());

        std::vector<SymmetricTensor<2, dim>> strain;
        strain.resize(input_data.solution_gradients.size());
        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {

            for (unsigned int i = 0; i < dim; ++i)
                strain[p][i][i] = input_data.solution_gradients[p][i][i];

            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = i + 1; j < dim; ++j)
                    strain[p][i][j] =
                        (input_data.solution_gradients[p][i][j] + input_data.solution_gradients[p][j][i]) / 2;
        }

        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
            AssertDimension(computed_quantities[p].size(), (Tensor<2, dim>::n_independent_components));
            for (unsigned int d = 0; d < dim; ++d)
            {
                for (unsigned int e = 0; e < dim; ++e)
                {
                    computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] =
                        40 * trace(strain[p]) * unit_symmetric_tensor<dim>()[d][e] + 2 * 40 * strain[p][d][e];
                }
            }
        }
    }
};

template <int dim> class ElasticProblem
{
  public:
    // Constructor
    ElasticProblem();

    void run();

  private:
    void setup_system(bool initial_step);
    void right_hand_side(std::vector<Tensor<1, dim>> &values);

    void assemble_system();
    void set_boundary_condition(bool initial_NR_step);
    void set_rhs();
    void solve_nonlinear_timestep();
    void test_system();
    void solve();

    void extract_stress();
    void gnuplot_out(const std::string filename) const;
    void output_results() const;
    void output_results_vtk() const;

    SymmetricTensor<4, dim> get_stiffness(const SymmetricTensor<2, dim> &strain_tensor);
    SymmetricTensor<2, dim> get_stress(const SymmetricTensor<2, dim> &strain_tensor);

    typedef enum
    {
        SHEAR,
        UNIAXIAL,
        FIXED,
    } BCTYPE;

    typedef enum
    {
        LINEAR,
        NONLINEAR,
    } SIMULATIONTYPE;

    BCTYPE BC_TYPE;
    SIMULATIONTYPE SIMULATION_TYPE;

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    const QGauss<dim> quadrature_formula;

    std::map<types::global_dof_index, double> boundary_values;

    AffineConstraints<double> constraints;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> tangent_matrix;

    Vector<double> solution_n;
    Vector<double> newton_update;
    Vector<double> system_rhs;

    static constexpr double lambda = 40;
    static constexpr double mu = 40;
    static constexpr double a =  40;
    static constexpr double b = -50;
    static constexpr double c = -30;

    int cycle;
};

// Constructor
template <int dim>
ElasticProblem<dim>::ElasticProblem()
    : BC_TYPE(UNIAXIAL), SIMULATION_TYPE(NONLINEAR), dof_handler(triangulation), fe(FE_Q<dim>(1), dim), quadrature_formula(fe.degree + 1)
{
}

template <int dim> void ElasticProblem<dim>::right_hand_side(std::vector<Tensor<1, dim>> &values)
{
    switch (BC_TYPE)
    {
    case SHEAR:
        break;  
    case UNIAXIAL:
        values[2][0] = 3.30632*0.5*cycle * 0; // 1.06667 3.30632
        values[4][0] = 3.30632*0.5*cycle * 0;
        break;
    case FIXED: 
        values[2][0] = 1.0*cycle; // 1.06667 3.30632
        values[4][0] = 1.0*cycle;
        break;
    }
}

// c_ijkl(e) = a*(delta_ik*delta_jl + delta_il*delta_jk) +
//             6*b*e_mm*delta_ij*delta_kl +
//             3/2*c*(delta_ik*e_jl + delta_jl*e_ik + delta_jk*e_il + delta_il*e_jk)
template <int dim>
SymmetricTensor<4, dim> ElasticProblem<dim>::get_stiffness(const SymmetricTensor<2, dim> &strain_tensor)
{
    SymmetricTensor<4, dim> tmp;
    switch (SIMULATION_TYPE)
    {
    case LINEAR:
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) + 
                                           ((i == l) && (j == k) ? mu : 0.0) +
                                           ((i == j) && (k == l) ? lambda : 0.0));
        break;
    case NONLINEAR:
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                            tmp[i][j][k][l] =
                                (((i == k) && (j == l) ? a + 0.75 * c * (strain_tensor[j][l] + strain_tensor[i][k])
                                                    : 0.0) +
                                 ((i == l) && (j == k) ? a + 0.75 * c * (strain_tensor[j][k] + strain_tensor[i][l])
                                                     : 0.0) +
                                 ((i == j) && (k == l) ? 3 * b * trace(strain_tensor) : 0.0));
        /*     tmp  = 2 * a * Physics::Elasticity::StandardTensors<dim>::IxI;
        tmp += 12* b * trace(strain_tensor) * Physics::Elasticity::StandardTensors<dim>::IxI;
        tmp += 3 * c * outer_product(identity_tensor<dim>()*strain_tensor, identity_tensor<dim>()*strain_tensor); */
        break;
    }
   
    return tmp;
}

template <int dim> 
SymmetricTensor<2, dim> ElasticProblem<dim>::get_stress(const SymmetricTensor<2, dim> &strain_tensor)
{
    SymmetricTensor<2, dim> tmp;

    switch (SIMULATION_TYPE)
    {
    case LINEAR:
        tmp = lambda * trace(strain_tensor) * unit_symmetric_tensor<dim>() + 2 * mu * strain_tensor;
        break;
    
    case NONLINEAR:
        tmp  = 2 * a * strain_tensor;
        // std::cout << "tmp: " << tmp << std::endl;
        tmp += 3 * b * trace(strain_tensor) * trace(strain_tensor) * unit_symmetric_tensor<dim>();    
        // std::cout << "tmp: " << tmp << std::endl;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int m = 0; m < dim; ++m)
                    tmp[i][j] += 3 * c * strain_tensor[i][m] * strain_tensor[j][m];    
        // std::cout << "tmp: " << tmp << std::endl;
        break;
    }
    
    return tmp;
}

template <int dim> void ElasticProblem<dim>::setup_system(bool initial_step)
{
    // Resize the solution and right hand side vectors
    if(initial_step)
    {
        // Allocate the memory for the DOFs and enumarate them
        dof_handler.distribute_dofs(fe);

        // Making use of sparsity pattern
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern.copy_from(dsp);

        solution_n.reinit(dof_handler.n_dofs());
        newton_update.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

    if (BC_TYPE == UNIAXIAL)
    {
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();

        for (; cell != endc; ++cell)
        {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v) // const auto v : cell->vertex_indices()
            {
              if (std::abs(cell->vertex(v)(0) + 0.5) < 1.0e-6 && std::abs(cell->vertex(v)(1) - 0.5) < 1.0e-6)
              {
                  const types::global_dof_index i = cell->vertex_dof_index(v, 0); 
                  constraints.add_line(i);
                  constraints.set_inhomogeneity(i, 0.0); 
                  std::cout << "Fixed vertex" << v << " is at " << cell->vertex(v) << std::endl;
                  std::cout << "DOF " << i << " is fixed" << std::endl;
              }
              if (std::abs(cell->vertex(v)(0) + 0.5) < 1.0e-6 && std::abs(cell->vertex(v)(1) + 0.5) < 1.0e-6)
              {
                  const types::global_dof_index i = cell->vertex_dof_index(v, 0); 
                  constraints.add_line(i);
                  constraints.set_inhomogeneity(i, 0.0); 
                  const types::global_dof_index j = cell->vertex_dof_index(v, 1); 
                  constraints.add_line(j);
                  constraints.set_inhomogeneity(j, 0.0); 
                  std::cout << "Fixed vertex" << v << " is at " << cell->vertex(v) << std::endl;
                  std::cout << "DOF " << i << " is fixed" << std::endl;
                  std::cout << "DOF " << j << " is fixed" << std::endl;
              }
            }
        }
        constraints.close();
    }

    // Resize the system matrix
    tangent_matrix.reinit(sparsity_pattern);
}

template <int dim> void ElasticProblem<dim>::assemble_system()
{
    // FE object
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    const FEValuesExtractors::Vector displacement(0);
    right_hand_side(rhs_values);

    // Loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);

        std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
        fe_values[displacement].get_function_symmetric_gradients(solution_n, strain_tensor);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const SymmetricTensor<2, dim> eps_phi_i = get_sym_grad(fe_values, i, q_point),
                                                  eps_phi_j = get_sym_grad(fe_values, j, q_point);

                    const SymmetricTensor<4, dim> stiffness = get_stiffness(strain_tensor[q_point]);
                    
                    // K_ij = int(eps_i * C * eps_j) dx
                    cell_matrix(i, j) += (eps_phi_i * stiffness * eps_phi_j) * fe_values.JxW(q_point);
                }

        // Loop over all quadrature points for RHS == 0 since
        // there is Neumann boundary condition & force applied
        for (const unsigned int i : fe_values.dof_indices())
        {
            const unsigned int component_i = fe.system_to_component_index(i).first;
            for (const unsigned int q_point : fe_values.quadrature_point_indices())
            {
                const SymmetricTensor<2, dim> eps_phi_i = get_sym_grad(fe_values, i, q_point);
                const SymmetricTensor<2, dim> stress    = get_stress(strain_tensor[q_point]);
                cell_rhs(i) += rhs_values[q_point][component_i] * fe_values.JxW(q_point) * 0.0;
                cell_rhs(i) += eps_phi_i * stress * fe_values.JxW(q_point);
            }
        }

        // Assembly
        cell->get_dof_indices(local_dof_indices);
        system_rhs.reinit(dof_handler.n_dofs());
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, tangent_matrix, system_rhs);
    }

    std::cout << "End of assemble_system()" << std::endl;
    std::cout << "RHS =" << system_rhs << std::endl;
    std::cout << "n_constraints = " << constraints.n_constraints() << std::endl;

}


template <int dim> void ElasticProblem<dim>::set_boundary_condition(bool initial_NR_step)
{   
    if (initial_NR_step)
    {
    // Set the boundary conditions
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    

    switch (BC_TYPE)
    {
    case SHEAR:
        // Right boundary, u_y = 0.01*L
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(
                dof_handler, boundary_id,
                Functions::ConstantFunction<dim>(std::vector<double>({0.0, 0.02 })), boundary_values,
                fe.component_mask(y_displacement));
        }
        // Right boundary, u_x = 0
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler, boundary_id, Functions::ZeroFunction<dim>(dim),
                                                     boundary_values, fe.component_mask(x_displacement));
        }
        // Left boundary, u_x = u_y = 0
        {
            const int boundary_id = 3;
            VectorTools::interpolate_boundary_values(dof_handler, boundary_id, Functions::ZeroFunction<dim>(dim),
                                                     boundary_values);
        }
        break;
    case FIXED:
        // Right boundary, u_x = 0.01*L
/*         {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(
                dof_handler, boundary_id,
                Functions::ConstantFunction<dim>(std::vector<double>({0.02 , 0.0})), boundary_values,
                fe.component_mask(x_displacement));
        } */
        // Left boundary, u_x = u_y = 0
        {
            const int boundary_id = 3;
            VectorTools::interpolate_boundary_values(dof_handler, boundary_id, Functions::ZeroFunction<dim>(dim),
                                                     boundary_values);
        }
        break;
    case UNIAXIAL:
        // Right boundary, u_x = 0.01*L
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(
                dof_handler, boundary_id,
                Functions::ConstantFunction<dim>(std::vector<double>({0.0025 , 0.0})), boundary_values,
                fe.component_mask(x_displacement));
        }   
        // Left boundary, u_x = 0
        {
            const int boundary_id = 3;
            VectorTools::interpolate_boundary_values(dof_handler, boundary_id, Functions::ZeroFunction<dim>(dim),
                                                     boundary_values, fe.component_mask(x_displacement));
        } 
        break;
      }
    }
    // Apply the boundary conditions
    MatrixTools::apply_boundary_values(boundary_values, 
                                       tangent_matrix, 
                                       newton_update, 
                                       system_rhs, 
                                       true);

    std::cout << "End of set_boundary_condition()" << std::endl;
    std::cout << "RHS =" << system_rhs << std::endl;
}

template <int dim> void ElasticProblem<dim>::set_rhs()
{
    // Set the boundary conditions
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    // Right boundary, u_x = 0.01*L
    {
        const int boundary_id = 1;
        VectorTools::interpolate_boundary_values(
            dof_handler, boundary_id,
            Functions::ConstantFunction<dim>(std::vector<double>({0.0025 + newton_update[6], 0.0 })), boundary_values,
                fe.component_mask(x_displacement));
    }
    {
    const int boundary_id = 3;
    VectorTools::interpolate_boundary_values(dof_handler, boundary_id, Functions::ZeroFunction<dim>(dim),
                                             boundary_values, fe.component_mask(x_displacement));
    }
    // Apply the boundary conditions
    MatrixTools::apply_boundary_values(boundary_values, 
                                       tangent_matrix, 
                                       newton_update, 
                                       system_rhs, 
                                       false);

    std::cout << "End of set_rhs()" << std::endl;
    std::cout << "RHS =" << system_rhs << std::endl;
}

template <int dim> void ElasticProblem<dim>::solve_nonlinear_timestep()
{

    double residual = 1e10;
    unsigned int newton_iteration = 0;

    Vector<double> initial_rhs(dof_handler.n_dofs());
    Vector<double> error_res(dof_handler.n_dofs());
    error_res = 0.0;

/*     newton_update[0] = 0.0;
    newton_update[1] = 0.0;
    newton_update[2] = 0.001;
    newton_update[3] = 0;
    newton_update[4] = 0.0;
    newton_update[5] = 0.001; // 005897657980254
    newton_update[6] = 0.001;
    newton_update[7] = 0.001; // */

    assemble_system();

    /* system_rhs.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs()); */
    bool initial_NR_step = true;

    set_boundary_condition(initial_NR_step);

    solve();
    solution_n.add(1.0, newton_update);
    assemble_system();
    set_rhs();
    solution_n.reinit(dof_handler.n_dofs());
/*     system_rhs[0] = 0;
    system_rhs[1] = 0;
    system_rhs[2] = 1.76429;
    system_rhs[3] = -0.209616; // 209616
    system_rhs[4] = 0;
    system_rhs[5] = 0.209616; //
    system_rhs[6] = 1.76429;
    system_rhs[7] = 0.209616; // */
    initial_rhs = system_rhs;
    while (newton_iteration < 20)
    {     
        solve();
        
        solution_n.add(1.0, newton_update);

        std::cout << "SLN =" << solution_n << std::endl;
        std::cout << "DNU =" << newton_update << std::endl;

        assemble_system();
        
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
        switch (BC_TYPE)
        {
            case UNIAXIAL:
            // i == 2 || i == 3 || i == 5 || i == 6  || i == 7 
                if (i == 3 || i == 5 || i == 7 ) // !constraints.is_constrained(i) &&
                  error_res[i] += system_rhs[i];
            break;
            case SHEAR:
                std::cout << "Shear not implemented yet" << std::endl;
            break;
            case FIXED:
                if ( i == 3 || i == 7 ) // !constraints.is_constrained(i) && 
                  error_res[i] += system_rhs[i];
            break;
        }

        residual = error_res.l2_norm();

        std::cout << "Newton iteration " << newton_iteration
                  << "-> residual: " << residual<< std::endl;
        std::cout << "--------------------------------------------" << std::endl;

        //system_rhs.reinit(dof_handler.n_dofs());

        //initial_NR_step = false;
        set_rhs();
        // system_rhs -= initial_rhs;
        // std::cout << "RHS =" << system_rhs << std::endl;
        newton_iteration++;
    }
}

template <int dim> void ElasticProblem<dim>::test_system()
{
    newton_update[0] = 0.0;
    newton_update[1] = 0.0;
    newton_update[2] = 0.05;
    newton_update[3] = 0;
    newton_update[4] = 0.0;
    newton_update[5] = 0.005897657980254;
    newton_update[6] = 0.05;
    newton_update[7] = 0.005897657980254; 

/*     newton_update[0] = 0.0;
    newton_update[1] = 0.0;
    newton_update[2] = 0.01;
    newton_update[3] = 0.0;
    newton_update[4] = 0.0;
    newton_update[5] = -0.00333334;
    newton_update[6] = 0.01;
    newton_update[7] = -0.00333334; */

    assemble_system();
    system_rhs.reinit(dof_handler.n_dofs());
/*  system_rhs[0] = 0.0;
    system_rhs[1] = 0.0;
    system_rhs[2] = 1.65316;
    system_rhs[3] = 0;
    system_rhs[4] = 0.0;
    system_rhs[5] = 0;
    system_rhs[6] = 1.65316;
    system_rhs[7] = 0;          */

/*     system_rhs[0] = 0.0;
    system_rhs[1] = 0.0;
    system_rhs[2] = 0.533333;
    system_rhs[3] = -2e-10;
    system_rhs[4] = 0.0;
    system_rhs[5] = 2e-10;
    system_rhs[6] = 0.533333;
    system_rhs[7] = 2e-10; */

    set_boundary_condition(true);

    std::cout << "SLN =" << solution_n << std::endl;
    std::cout << "DNU =" << newton_update << std::endl;

    solve();

    assemble_system();

    std::cout << "SLN =" << solution_n << std::endl;
    std::cout << "DNU =" << newton_update << std::endl;

}
// Output DOF numbering in gnuplot format (step-2)
template <int dim> void ElasticProblem<dim>::gnuplot_out(const std::string filename) const
{
    std::ofstream out(filename);
    out << "plot '-' using 1:2 with lines, "
        << "'-' with labels point pt 2 offset 1,1" << std::endl;
    GridOut().write_gnuplot(triangulation, out);
    out << "e" << std::endl;

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);
    DoFTools::write_gnuplot_dof_support_point_info(out, support_points);
    out << "e" << std::endl;
}

// CG solver without preconditioner
template <int dim> void ElasticProblem<dim>::solve()
{
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(tangent_matrix);
    A_direct.vmult(newton_update, system_rhs);
}
// Output results in vtk format
template <int dim> void ElasticProblem<dim>::output_results() const
{
    StrainPostprocessor<dim> strain_postprocessor;
    StressPostprocessor<dim> stress_postprocessor;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
    {
    case 1:
        solution_names.emplace_back("displacement");
        break;
    case 2:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        break;
    case 3:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        solution_names.emplace_back("z_displacement");
        break;
    default:
        Assert(false, ExcNotImplemented());
    }
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(solution_n, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
    data_out.add_data_vector(solution_n, strain_postprocessor);
    data_out.add_data_vector(solution_n, stress_postprocessor);
    // data_out.build_patches ();

    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
        soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(1, dof_handler, soln);
    data_out.build_patches(q_mapping, 1);

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
}

// Output results in vtu format
template <int dim> void ElasticProblem<dim>::output_results_vtk() const
{
    gnuplot_out("gnuplot.gpl");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
    {
    case 1:
        solution_names.emplace_back("displacement");
        break;
    case 2:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        break;
    case 3:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        solution_names.emplace_back("z_displacement");
        break;
    default:
        Assert(false, ExcNotImplemented());
    }

    data_out.add_data_vector(solution_n, solution_names);
    // data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");

    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
        soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(1, dof_handler, soln);
    data_out.build_patches(q_mapping, 1);

    data_out.write_vtk(output);
}

template <int dim> void ElasticProblem<dim>::extract_stress()
{
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::fstream out("stress-" + std::to_string(cycle) + ".txt", std::ios::out);

    std::vector<std::vector<Tensor<1, dim>>> displacement_grads(quadrature_formula.size(),
                                                                std::vector<Tensor<1, dim>>(dim));

    const FEValuesExtractors::Vector displacement(0);

    out << "get_sym_grad(displacement_grads[q_point] = " << std::endl;
    out << "strain_tensor[q_point] = " << std::endl;
    out << "stress = " << std::endl;
    out << "get_stress(strain_tensor[q_point])= " << std::endl;

    for (auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(solution_n, displacement_grads);
        std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
        fe_values[displacement].get_function_symmetric_gradients(solution_n, strain_tensor);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            out << "q_point = " << q_point << std::endl;
            const SymmetricTensor<4, dim> stiffness = get_stiffness(strain_tensor[q_point]);
            const SymmetricTensor<2, dim> stress =
                ((stiffness * strain_tensor[q_point])); // get_sym_grad(displacement_grads[q_point])

            out << get_sym_grad(displacement_grads[q_point]) << std::endl;
            out << strain_tensor[q_point] << std::endl;
            out << stress << std::endl;
            out << get_stress(strain_tensor[q_point]) << std::endl;
        }
    }
}

template <int dim> void ElasticProblem<dim>::run()
{
    // Create the triangulation
    GridGenerator::hyper_cube(triangulation, -0.5, 0.5);

//#if DEBUG
//    SymmetricTensor<2, dim> strain_tensor;
//    strain_tensor[0][0] = 0.05 ;
//    strain_tensor[0][1] = 0.0; //-0.00333087 ;
//    strain_tensor[1][1] = 0.0058976579802540; //-0.0182002;
//    std::cout << strain_tensor << std::endl;
//    SymmetricTensor<4, dim> stiffness = get_stiffness(strain_tensor);
//    std::cout << stiffness << std::endl;
//
//    SymmetricTensor<2, dim> stress = get_stress(strain_tensor);
//    std::cout << stress << std::endl;
//    return;
//#endif
    // Shift the top right corner node
    //
    // for (const auto &cell : triangulation.active_cell_iterators())
    // {
    //   for (unsigned int i=0; i<GeometryInfo<2>::vertices_per_cell; ++i)
    //     {
    //       Point<2> &v = cell->vertex(i);
    //       if ((std::abs(v(0)-1)<1e-5) && (std::abs(v(1)-1)<1e-5))
    //         v(0) += 1;
    //         v(1) += 1;
    //     }
    // }

    // Set the boundary ids
    triangulation.begin_active()->face(1)->set_boundary_id(1);
    triangulation.begin_active()->face(0)->set_boundary_id(3);
    triangulation.begin_active()->face(3)->set_boundary_id(1);
    // Refine the mesh
    // triangulation.refine_global(3);
    std::cout << "   Number of active cells:       " << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    cycle = 1;
    bool initial_step = true;

    for (unsigned int i = 0; i < 2; ++i)
    {
        setup_system(initial_step);
        std::cout << "----------------------------------------------------" << std::endl
                  << "Cycle\t" << cycle << std::endl
                  << "----------------------------------------------------" << std::endl;

        solve_nonlinear_timestep();
        // test_system();
        extract_stress();
        output_results();
        output_results_vtk();
        initial_step = false;
        cycle++;
    }
}
} // namespace Project

int main()
{
    try
    {
        Project::ElasticProblem<2> elastic_problem_2d;
        elastic_problem_2d.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}