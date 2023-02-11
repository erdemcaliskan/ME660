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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

// Solver includes
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
 
// Grid includes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

// DoF includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// FE includes
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_eulerian.h>

// Numerics includes
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
// C++ includes
#include <fstream>
#include <iostream>
 
namespace Project
{
  using namespace dealii;
 
  // Auxiliary function to compute the stress tensor from the displacement (step-18)

  // Get the 4x4x4x4 stiffness matrix for the given material constants 
  // C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
  template <int dim>
  SymmetricTensor<4, dim> get_stiffness(const double lambda,
                                        const double mu)
  {
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                               ((i == l) && (j == k) ? mu : 0.0) +
                               ((i == j) && (k == l) ? lambda : 0.0));
    return tmp;
  }
 
  // Get the strain tensor from the displacement gradient of the solution
  // e_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
  template <int dim>
  inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                            const unsigned int   shape_func,
                                            const unsigned int   q_point)
  {
    SymmetricTensor<2, dim> tmp;
 
    for (unsigned int i = 0; i < dim; ++i)
      tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];
 
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
          2;
 
    return tmp;
  }

  // Get the strain tensor from the displacement gradient
  // e_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
  template <int dim>
  inline SymmetricTensor<2, dim>
  get_strain(const std::vector<Tensor<1, dim>> &grad)
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
  
  template <int dim>
  class ElasticProblem
  {
  public:
    // Constructor
    ElasticProblem();

    // Run the program
    void run();
 
  private:

    // Handiling DOFs 
    void setup_system();

    // Assemble the system matrix and right hand side
    void assemble_system();

    // Solve the system
    void solve();

    // Compute the stress tensor at each quadrature point
    void extract_stress();

    // Output the DOFs
    void gnuplot_out(const std::string filename) const;

    // Output the solution
    void output_results() const;

    // Mesh object
    Triangulation<dim> triangulation;
    
    // DOF object
    DoFHandler<dim>    dof_handler;
 
    // FE object
    FESystem<dim> fe;
    const QGauss<dim> quadrature_formula;
    
    // Constraint object
    AffineConstraints<double> constraints;
 
    // Sparsity pattern and system matrix
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    // Solution and right hand side
    Vector<double> solution;
    Vector<double> system_rhs;

    // Stiffness matrix
    static const SymmetricTensor<4, dim> stiffness;

    // Material constants
    static constexpr double lambda = 40;
    static constexpr double mu = 40;
  };
 
  // Constructor
  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , quadrature_formula(fe.degree + 1)
  {}
  
  // Stiffness matrix
  template <int dim>
  const SymmetricTensor<4, dim> ElasticProblem<dim>::stiffness =
    get_stiffness<dim>(lambda,
                       mu    );
 
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    // Allocate the memory for the DOFs and enumarate them
    dof_handler.distribute_dofs(fe);

    // Resize the solution and right hand side vectors
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Making use of sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
 
    // Resize the system matrix
    system_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    // FE object
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    // Loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
 
        fe_values.reinit(cell);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                const SymmetricTensor<2, dim>
                  eps_phi_i = get_strain(fe_values, i, q_point),
                  eps_phi_j = get_strain(fe_values, j, q_point);

                // K_ij = int(eps_i * C * eps_j) dx
                cell_matrix(i, j) += (eps_phi_i *            
                                      stiffness * 
                                      eps_phi_j              
                                      ) *                    
                                     fe_values.JxW(q_point); 
              }

        // Loop over all quadrature points for RHS == 0 since
        // there is Neumann boundary condition & force applied
        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
              cell_rhs(i) += fe_values.shape_value(i, q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point) * 0.0 ;
          }

        // Assembly
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    // Set the boundary conditions
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    std::map<types::global_dof_index, double> boundary_values;

    // Right boundary, u = 0.01*L
    {
      const int boundary_id = 1;

      VectorTools::interpolate_boundary_values(
        dof_handler,
        boundary_id,
        Functions::ConstantFunction<dim> (std::vector<double>({0.02, 0.})),
        boundary_values,
        fe.component_mask(x_displacement));
    }

    // Left boundary, u = 0
    {
      const int boundary_id = 3;

      VectorTools::interpolate_boundary_values(
        dof_handler,
        boundary_id,
        Functions::ZeroFunction<dim>(dim),
        boundary_values,
        fe.component_mask(x_displacement));
    }

    // Apply the boundary conditions
    MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, true);
  }
 
  // Output DOF numbering in gnuplot format (step-2)
  template <int dim>
  void ElasticProblem<dim>::gnuplot_out(const std::string filename) const
  {
    std::ofstream out(filename);
    out << "plot '-' using 1:2 with lines, "
        << "'-' with labels point pt 2 offset 1,1"
        << std::endl;
    GridOut().write_gnuplot (triangulation, out);
    out << "e" << std::endl;
    
    std::map<types::global_dof_index, Point<dim> > support_points;
    DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
                                          dof_handler,
                                          support_points);
    DoFTools::write_gnuplot_dof_support_point_info(out,
                                                   support_points);
    out << "e" << std::endl;
  }
 
  // CG solver without preconditioner
  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  }

  // Output results in vtu format
  template <int dim>
  void ElasticProblem<dim>::output_results() const
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
 
    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();
 
    std::ofstream output("solution.vtk");
    data_out.write_vtk(output);
  } 
  
  // Extract stress from the solution for each quadrature point
  template<int dim>
  void ElasticProblem<dim>::extract_stress()
  {
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::fstream out("stress.txt", std::ios::out);

    std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values.get_function_gradients(solution, displacement_grads);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const SymmetricTensor<2, dim> stress = 
          ((stiffness *
                get_strain(displacement_grads[q_point])));

        out << stress << std::endl;
        }
    }
  }

  template <int dim>
  void ElasticProblem<dim>::run()
  {
    // Create the triangulation
    GridGenerator::hyper_cube(triangulation, -1, 1);
    
    // Set the boundary ids
    triangulation.begin_active()->face(1)->set_boundary_id(1);
    triangulation.begin_active()->face(0)->set_boundary_id(3);
    
    // Refine the mesh
    // triangulation.refine_global(5);

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    
    setup_system();
 
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
 
    assemble_system();
    solve();
    extract_stress();
    output_results();

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
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}