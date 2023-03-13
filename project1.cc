/* ---------------------------------------------------------------------
 *
 * This program solves the linear elasticity problem on a rectangular
 * domain. It is based on step-8, but also uses step 18 and 44. 
 * 
 * https://www.dealii.org/current/doxygen/deal.II/step_8.html
 * https://www.dealii.org/current/doxygen/deal.II/step_18.html
 * https://www.dealii.org/current/doxygen/deal.II/step_44.html
 * 
 * ---------------------------------------------------------------------
 * 
 * Tested with linear Cijlk and it is working. But I couldn't figure out
 * why the residual is not changing after the Newton step.
 * 
 * Future work:
 * 
 * Input file will be added to read the material constants and other properties.
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
#include <deal.II/lac/sparse_direct.h>

// Grid includes
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
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

  // Get the 4x4x4x4 stiffness matrix for the given material constants (NOT IN USE)
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
  inline SymmetricTensor<2, dim> get_sym_grad(const FEValues<dim> &fe_values,
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
  get_sym_grad(const std::vector<Tensor<1, dim>> &grad)
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
      void setup_system();

      void assemble_system(bool update_stiffness);
      void right_hand_side(std::vector<Tensor<1, dim>> &values);
      void solve_nonlinear_timestep();
      void solve();
      void extract_stress();
      void gnuplot_out

      (const std::string filename) const;
      void output_results() const;

      Triangulation<dim> triangulation;
      DoFHandler<dim> dof_handler;

      FESystem<dim> fe;
      const QGauss<dim> quadrature_formula;

      AffineConstraints<double> constraints;

      SparsityPattern sparsity_pattern;
      SparseMatrix<double> tangent_matrix;

      Vector<double> solution_n;
      Vector<double> newton_update;
      Vector<double> system_rhs;
      SymmetricTensor<4, dim> get_stiffness(const SymmetricTensor<2, dim> &strain_tensor);
      SymmetricTensor<4, dim> get_stiffness_linear(const SymmetricTensor<2, dim> &strain_tensor);
      SymmetricTensor<2, dim> get_stress(const SymmetricTensor<2, dim> &strain_tensor);
      SymmetricTensor<2, dim> get_stress_linear(const SymmetricTensor<2, dim> &strain_tensor);
      // Stiffness matrix
      SymmetricTensor<4, dim> stiffness;

      // Material constants
      static constexpr double lambda = 40;
      static constexpr double mu = 40;

      static constexpr double a =  40;
      static constexpr double b = -50;
      static constexpr double c = -30;

      int cycle;
      double force;
  };
 
  // Constructor
  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , quadrature_formula(fe.degree + 1)
  {}
  
  template <int dim>
  void ElasticProblem<dim>::right_hand_side(std::vector<Tensor<1, dim>> &  values)
  {
    for (unsigned int point_n = 0; point_n < 4; ++point_n)
      {
        values[point_n][0] = force*0.0;
      }
  }

  // Get the 4x4x4x4 stiffness matrix for the given material constants (NOT IN USE)
  // C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
  template <int dim>
  SymmetricTensor<4, dim> ElasticProblem<dim>::
                          get_stiffness_linear(const SymmetricTensor<2, dim> &strain_tensor)
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

  // c_ijkl(e) = a(delta_ik*delta_jl + delta_il*delta_jk) + 
  //             6*b*e_mm*delta_ij*delta_kl + 
  //             3/2*c*(delta_ik*e_jl + delta_jl*e_ik + delta_jk*e_il + delta_il*e_jk)
  template <int dim>
  SymmetricTensor<4, dim> ElasticProblem<dim>::
                          get_stiffness(const SymmetricTensor<2, dim> &strain_tensor)
  {
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            for (unsigned int m = 0; m < dim; ++m)
              tmp[i][j][k][l] = (((i == k) && (j == l) ? a + 3/2*c*(strain_tensor[j][l] + strain_tensor[i][k]) : 0.0) +
                                 ((i == l) && (j == k) ? a + 3/2*c*(strain_tensor[j][k] + strain_tensor[i][l]) : 0.0) +
                                 ((i == j) && (k == l) ? 6*b*strain_tensor[m][m] : 0.0));
    return tmp;     
  }

  template <int dim>
  SymmetricTensor<2, dim> ElasticProblem<dim>::get_stress_linear(const SymmetricTensor<2, dim> &strain_tensor)
  {
    SymmetricTensor<2, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int m = 0; m < dim; ++m)
            tmp[i][j] = ((i == j)  ? lambda*strain_tensor[m][m] 
                                   : 2*mu*strain_tensor[i][j] );
    return tmp;   
  }

  template <int dim>
  SymmetricTensor<2, dim> ElasticProblem<dim>::get_stress(const SymmetricTensor<2, dim> &strain_tensor)
  {
    SymmetricTensor<2, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int m = 0; m < dim; ++m)
          for (unsigned int n = 0; n < dim; ++n)
            tmp[i][j] = ((i == j)  ? 3*b*strain_tensor[m][m]*strain_tensor[n][n] 
                                   : 2*a*strain_tensor[i][j] + 
                                     3*c*strain_tensor[i][m]*strain_tensor[j][m]);
    return tmp;   
  }
 
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    // Allocate the memory for the DOFs and enumarate them
    dof_handler.distribute_dofs(fe);

    // Resize the solution and right hand side vectors
    newton_update.reinit(dof_handler.n_dofs());
    solution_n.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    std::vector<bool> boundary_dofs (dof_handler.n_dofs(), false);
    DoFTools::extract_boundary_dofs (dof_handler,
                                    ComponentMask(),
                                    boundary_dofs);

    const unsigned int first_boundary_dof
      = std::distance (boundary_dofs.begin(),
                      std::find (boundary_dofs.begin(),
                                 boundary_dofs.end(),
                                 true));
    constraints.clear ();
    constraints.add_line (first_boundary_dof);
    constraints.add_line (first_boundary_dof + 1);
    for (unsigned int i=first_boundary_dof+1; i<dof_handler.n_dofs(); ++i)
      if (boundary_dofs[i] == true)
        constraints.add_entry (first_boundary_dof,
                                          i, 0);
    constraints.close ();

    // Making use of sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
 
    // Resize the system matrix
    tangent_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system(bool update_stiffness)
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

    const FEValuesExtractors::Vector displacement(0);

    // Loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
 
        fe_values.reinit(cell);
        if (update_stiffness)
        {     
        right_hand_side(rhs_values);

        std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
        fe_values[displacement].get_function_symmetric_gradients(solution_n, strain_tensor);
        //std::cout << std::endl << "strain_tensor[0] = " << strain_tensor[0] << std::endl;
        //std::cout << "strain_tensor[1] = " << strain_tensor[1] << std::endl;
        //std::cout << "strain_tensor[2] = " << strain_tensor[2] << std::endl;
        //std::cout << "strain_tensor[3] = " << strain_tensor[3] << std::endl;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                
                const SymmetricTensor<2, dim>
                  eps_phi_i = get_sym_grad(fe_values, i, q_point),
                  eps_phi_j = get_sym_grad(fe_values, j, q_point);
                  stiffness = get_stiffness_linear(strain_tensor[q_point]);

                const SymmetricTensor<2, dim> stress = get_stress_linear(strain_tensor[q_point]);
                // K_ij = int(eps_i * C * eps_j) dx
                cell_matrix(i, j) += (eps_phi_i *            
                                      stiffness * 
                                      eps_phi_j              
                                      ) *                    
                                     fe_values.JxW(q_point); 
                cell_rhs(i) -= eps_phi_i * stress; 
              }
        }
        // Loop over all quadrature points for RHS == 0 since
        // there is Neumann boundary condition & force applied

        std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
          quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

      
        for (const unsigned int i : fe_values.dof_indices())
          {
            std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
            fe_values[displacement].get_function_symmetric_gradients(solution_n, strain_tensor);
            
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
                 {

                  stiffness = get_stiffness(strain_tensor[q_point]);
                  const SymmetricTensor<2, dim> stress = get_stress_linear(strain_tensor[q_point]);
                  if(update_stiffness)
                  {
                  cell_rhs(i) += (fe_values.shape_value(i, q_point) *
                             (rhs_values[q_point][component_i]  ))* //   - stress * strain_tensor[q_point]
                             fe_values.JxW(q_point) ;
                  
                  }
                  
                  //else
                  //cell_rhs(i) -= (stress * strain_tensor[q_point])* //   
                  //           fe_values.JxW(q_point);
                  //
                  }
          }
        // std::cout<< std::endl << "cell_rhs = " << cell_rhs << std::endl;
        // Assembly
        cell->get_dof_indices(local_dof_indices);
        //system_rhs.reinit(dof_handler.n_dofs());
        if (update_stiffness)
          constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, tangent_matrix, system_rhs);
        else
        {
          // system_rhs.reinit(dof_handler.n_dofs());
          // cell_rhs *= -1;
          constraints.distribute_local_to_global(
          cell_rhs, local_dof_indices, system_rhs);
        }         
      }

    // Set the boundary conditions
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    std::map<types::global_dof_index, double> boundary_values;

    if (update_stiffness)
    {
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
              fe.component_mask(x_displacement) | fe.component_mask(y_displacement));
          }

          // Apply the boundary conditions
          MatrixTools::apply_boundary_values(boundary_values, 
                                           tangent_matrix, 
                                           newton_update, 
                                           system_rhs, 
                                           true);
      }
    
  }

  template <int dim>
  void ElasticProblem<dim>::solve_nonlinear_timestep()
  {
    double residual = 1e10;
    unsigned int newton_iteration = 0;
    assemble_system(true);
    while (newton_iteration < 10)
    {
      solve();
      // constraints.distribute(newton_update);
      assemble_system(true);
      
      //solution.add(1.0, newton_update);
      solution_n += newton_update;
      residual = system_rhs.l2_norm();
      std::cout << "Newton iteration " << newton_iteration
                << "-> residual: " << residual << std::endl;
      if(residual < 1e-6)
        break;
      //newton_update.reinit(dof_handler.n_dofs());
      newton_iteration++;
    }
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
    solver.solve(tangent_matrix, newton_update, system_rhs, PreconditionIdentity());  
    /* SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix);
            A_direct.vmult(newton_update, system_rhs); */
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
 
    data_out.add_data_vector(solution_n, solution_names);
    data_out.build_patches();
 
    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
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

    std::fstream out("stress-" + std::to_string(cycle) + ".txt", std::ios::out);

    const FEValuesExtractors::Vector displacement(0);
    

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      std::vector<SymmetricTensor<2, dim>> strain_tensor(n_q_points);
      fe_values[displacement].get_function_symmetric_gradients(solution_n, strain_tensor);
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          
          out << get_stress_linear(strain_tensor[q_point]) << std::endl;
        }
    }
  }

  template <int dim>
  void ElasticProblem<dim>::run()
  {
    // Create the triangulation
    GridGenerator::hyper_cube(triangulation, -1, 1);

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
    
    // Refine the mesh
    // triangulation.refine_global(4);

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    
    setup_system();
 
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    cycle = 0;
    force = 100;
    for(unsigned int i=0; i<10; ++i)
    {
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Cycle\t" << cycle 
                << std::endl
                << "----------------------------------------------------"
                << std::endl;

      solve_nonlinear_timestep();
      extract_stress();
      output_results();
      cycle++;
      force += 100;
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