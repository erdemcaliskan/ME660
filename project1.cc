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
#include <deal.II/lac/sparse_direct.h>

// Solver includes
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/affine_constraints.h>
 
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

  // get tr(e)
  template <int dim>
  inline double get_e_trace(const SymmetricTensor<2, dim> &strain)
  {
    double trace = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
      trace += strain[i][i];
    return trace;
  }

  // c_ijkl(e) = a(delta_ik*delta_jl + delta_il*delta_jk) + 
  //             6*b*e_mm*delta_ij*delta_kl + 
  //             3/2*c*(delta_ik*e_jl + delta_jl*e_ik + delta_jk*e_il + delta_il*e_jk)
   template <int dim>
  SymmetricTensor<4, dim> get_stiffness(const FEValues<dim> &fe_values,
                                        const unsigned int   shape_func,
                                        const unsigned int   q_point,
                                        const double a,
                                        const double b,
                                        const double c)
  {
    const SymmetricTensor<2, dim> strain = get_sym_grad(fe_values,shape_func,q_point);
    const double tr_e = get_e_trace(strain);
    SymmetricTensor<4, dim> tmp;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp[i][j][k][l] = (((i == k) && (j == l) ? a + 3/2*c*(strain[j][l] + strain[i][k]) : 0.0) +
                               ((i == l) && (j == k) ? a + 3/2*c*(strain[j][k] + strain[i][l]) : 0.0) +
                               ((i == j) && (k == l) ? 6*b*tr_e : 0.0));
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
  void right_hand_side(const double applied_stress,
                       std::vector<Tensor<1, dim>> &  values)
  {
    for (unsigned int point_n = 0; point_n < 4; ++point_n)
      {
        values[point_n][0] = applied_stress;
      }
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
    void setup_system(const bool initial_step);

    // Assemble the system matrix and right hand side
    void assemble_system(double applied_stress);
    void solve_nonlinear_timestep(Vector<double> &solution_delta, double applied_stress);
    // Solve the system
    void solve(Vector<double> &newton_update);

    Vector<double> ElasticProblem<dim>::get_total_solution(const Vector<double> &solution_delta) const;

    // Compute the stress tensor at each quadrature point
    void extract_stress(const unsigned int cycle);

    // Output the DOFs
    void gnuplot_out(const std::string filename) const;

    // Output the solution
    void output_results(const unsigned int cycle) const;

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
    SparseMatrix<double> tangent_matrix;

    // Solution and right hand side
    Vector<double> solution_n;
    Vector<double> system_rhs;
    Vector<double> newton_update;
    Vector<double> current_solution;

    // Stiffness matrix
    SymmetricTensor<4, dim> stiffness;

    // Material constants
    static constexpr double lambda = 40;
    static constexpr double mu     = 40;
    static constexpr double a = 40;
    static constexpr double b = -50;
    static constexpr double c = -30;

        struct Errors
    {
      Errors() : norm(1.0)
      {}
 
      void reset()
      {
        norm = 1.0;
      }
      void normalize(const Errors &rhs)
      {
        norm /= rhs.norm;
      }
 
      double norm;
    };
 
    Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm;
 
    void get_error_residual(Errors &error_residual);
 
    void get_error_update(const Vector<double> &newton_update,
                          Errors &              error_update);

  };
 
  // Constructor
  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , quadrature_formula(fe.degree + 1)
  {}

  template <int dim>
  void ElasticProblem<dim>::get_error_residual(Errors &error_residual)
  {
    Vector<double> error_res(dof_handler.n_dofs());
 
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);
 
    error_residual.norm = error_res.l2_norm();
  }

  template <int dim>
  void ElasticProblem<dim>::get_error_update(const Vector<double> &newton_update,
                                             Errors &              error_update)
  {
    Vector<double> error_ud(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);
 
    error_update.norm = error_ud.l2_norm();
  }

  template <int dim>
  Vector<double> ElasticProblem<dim>::get_total_solution(
    const Vector<double> &solution_delta) const
  {
    Vector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
  }
/*   // Stiffness matrix
  template <int dim>
  const SymmetricTensor<4, dim> ElasticProblem<dim>::stiffness =
    get_stiffness<dim>(lambda,
                       mu    ); */
 
  template <int dim>
  void ElasticProblem<dim>::setup_system(const bool initial_step)
  {
      if (initial_step)
    {
      // Allocate the memory for the DOFs and enumarate them
      dof_handler.distribute_dofs(fe);

      // Resize the solution and right hand side vectors
      solution_n.reinit(dof_handler.n_dofs());
    }
    
    newton_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // u_y = 0 at bottom left node
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

    tangent_matrix.clear();
    tangent_matrix.reinit(sparsity_pattern);
    // Resize the system matrix
    system_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system(double applied_stress)
  {
    // QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
    // const unsigned int n_face_q_points = face_quadrature_formula.size();

    tangent_matrix = 0.0;

    // FE object
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    // FEFaceValues<dim> fe_face_values(fe,
    //                                 face_quadrature_formula,
    //                                 update_values | update_quadrature_points |
    //                                   update_normal_vectors |
    //                                   update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
    std::vector<Tensor<1, dim>> rhs_values(n_q_points);
    
    // Tensor<1, dim> pressure;
    // pressure[0] = 10;

    // Loop over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
 
        fe_values.reinit(cell);

        right_hand_side(applied_stress, rhs_values);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                const SymmetricTensor<2, dim>
                  eps_phi_i = get_sym_grad(fe_values, i, q_point),
                  eps_phi_j = get_sym_grad(fe_values, j, q_point);
                  stiffness = get_stiffness(fe_values, j, q_point, a , b ,c);
                // K_ij = int(eps_i * C * eps_j) dx
                cell_matrix(i, j) += (eps_phi_i *            
                                      stiffness * 
                                      eps_phi_j              
                                      ) *                    
                                     fe_values.JxW(q_point);
              }

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;
 
            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
              cell_rhs(i) = fe_values.shape_value(i, q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point);
          }
        
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, tangent_matrix, system_rhs);

          std::cout << "system_rhs = " << system_rhs << std::endl;
      }

    // Set the boundary conditions
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    std::map<types::global_dof_index, double> boundary_values;

    // Right boundary, u = 0.01*L
/*     {
      const int boundary_id = 1;

      VectorTools::interpolate_boundary_values(
        dof_handler,
        boundary_id,
        Functions::ConstantFunction<dim> (std::vector<double>({0.02, 0.})),
        boundary_values,
        fe.component_mask(x_displacement));
    } */

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
    boundary_values, tangent_matrix, newton_update, system_rhs, true);
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
 
  template <int dim>
  void ElasticProblem<dim>::solve_nonlinear_timestep(Vector<double> &solution_delta, double applied_stress)
  {
    Vector<double> solution_total(solution_n);
    Vector<double> newton_update(dof_handler.n_dofs());
 
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();
 
    unsigned int newton_iteration = 0;
    for (; newton_iteration < 100; ++newton_iteration)
      {
        std::cout << ' ' << std::setw(2) << newton_iteration << ' '
                  << std::flush;
 
        // make_constraints(newton_iteration);
        assemble_system(applied_stress);
 
        get_error_residual(error_residual);
        if (newton_iteration == 0)
          error_residual_0 = error_residual;
 
        error_residual_norm = error_residual;
        error_residual_norm.normalize(error_residual_0);
 
        if (newton_iteration > 0 && error_update_norm.norm <= 1e-16 &&
            error_residual_norm.norm <= 1e-16)
          {
            std::cout << " CONVERGED! " << std::endl;
             
            break;
          }
 
        //const std::pair<unsigned int, double> lin_solver_output =
        solve(newton_update);
 
        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;
 
        error_update_norm = error_update;
        error_update_norm.normalize(error_update_0);

        solution_delta += newton_update;
        
        Vector<double> solution_total(get_total_solution(solution_delta));

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < fe.; ++i)
          {
            const unsigned int component_i =
                fe.system_to_component_index(i).first;
            solution_total(local_dof_indices[i]) += newton_update(local_dof_indices[i]);
          }
        }
        //update_qph_incremental(solution_delta);
 
        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << error_residual_norm.norm << "  " << error_update_norm.norm << std::endl;
      }

    // solution_total += solution_delta;

    AssertThrow(newton_iteration < 100,
                ExcMessage("No convergence in nonlinear solver!"));
  } 

  // CG solver without preconditioner
  template <int dim>
  void ElasticProblem<dim>::solve(Vector<double> &newton_update)
  {
    Vector<double> A(dof_handler.n_dofs());

    unsigned int lin_it  = 0;
    double       lin_res = 0.0;

    SolverControl            solver_control(100, system_rhs.l2_norm() * 1e-6);
    std::cout << "system_rhs.l2_norm() = " << system_rhs.l2_norm() << std::endl;

    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(tangent_matrix, 1.2);

    //solver.solve(tangent_matrix, newton_update, system_rhs, PreconditionIdentity());
    SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix);
            A_direct.vmult(newton_update, system_rhs);
/*     lin_it  = 1; //solver_control.last_step();
    lin_res = 1; //solver_control.last_value();

    std::cout << "lin_it = " << lin_it << "\t";
    std::cout << "lin_res = " << lin_res << std::endl; */

    // solution.add(1, newton_update);
    constraints.distribute(newton_update);
  }

  // Output results in vtu format
  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
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
  void ElasticProblem<dim>::extract_stress(const unsigned int cycle)
  {
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::fstream out("stress-" + std::to_string(cycle) + ".txt", std::ios::out);

    std::vector<std::vector<Tensor<1, dim>>> displacement_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values.get_function_gradients(solution_n, displacement_grads);

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
    const int max_cycle = 100;
    double applied_stress = 6.0/max_cycle;
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
    setup_system(true);    

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;

    Vector<double> solution_delta(dof_handler.n_dofs());

     for (unsigned int cycle = 0; cycle < max_cycle; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;
      
      assemble_system(applied_stress);
      solve_nonlinear_timestep(solution_delta, applied_stress);
      setup_system(true);
      extract_stress(cycle);
      output_results(cycle);
      applied_stress += 6.0/max_cycle; 
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