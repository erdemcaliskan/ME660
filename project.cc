/* ---------------------------------------------------------------------
 *
 * This program solves the linear elasticity problem on a rectangular
 * domain. It is based on step-8, but also uses step 18 and 44.
 * 
 * Also see: 
 * The 'Quasi-Static Finite-Strain Compressible Elasticity' code gallery program 
 *
 * https://www.dealii.org/current/doxygen/deal.II/step_8.html
 * https://www.dealii.org/current/doxygen/deal.II/step_18.html
 * https://www.dealii.org/current/doxygen/deal.II/step_44.html
 * https://www.dealii.org/current/doxygen/deal.II/code_gallery_Quasi_static_Finite_strain_Compressible_Elasticity.html
 *
 * Author: Erdem Caliskan, 2023
 * ME 660, Spring 2023
 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/config.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>
#include <memory>

namespace Finite_strain_Neo_Hookian
{
using namespace dealii;

namespace Parameters
{

struct AssemblyMethod
{
    unsigned int automatic_differentiation_order;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void AssemblyMethod::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Assembly method");
    {
        prm.declare_entry(
            "Automatic differentiation order", "0", Patterns::Integer(0, 2),
            "The automatic differentiation order to be used in the assembly of the linear system.\n"
            "# Order = 0: Both the residual and linearisation are computed manually.\n"
            "# Order = 1: The residual is computed manually but the linearisation is performed using AD.\n"
            "# Order = 2: Both the residual and linearisation are computed using AD.");
    }
    prm.leave_subsection();
}

void AssemblyMethod::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Assembly method");
    {
        automatic_differentiation_order = prm.get_integer("Automatic differentiation order");
    }
    prm.leave_subsection();
}

struct FESystem
{
    unsigned int poly_degree;
    unsigned int quad_order;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void FESystem::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        prm.declare_entry("Polynomial degree", "2", Patterns::Integer(0), "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "3", Patterns::Integer(0), "Gauss quadrature order");
    }
    prm.leave_subsection();
}

void FESystem::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
    }
    prm.leave_subsection();
}

struct Geometry
{
    std::string shape;
    unsigned int elements_per_edge;
    double scale;
    double load_scale;
    std::string load_case;
    std::string load_type;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Geometry::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry");
    {
        prm.declare_entry("Shape", "square", Patterns::Selection("square|transformed"), 
                          "Shape (square|transformed)");

        prm.declare_entry("Elements per edge", "32", Patterns::Integer(0),
                          "Number of elements per long edge of the beam");

        prm.declare_entry("Grid scale", "1e-3", Patterns::Double(0.0), "Global grid scaling factor");

        prm.declare_entry("Load scale", "1.0", Patterns::Double(0.0), "Load scaling factor");

        prm.declare_entry("Load case", "pinned tension", Patterns::Selection("pinned tension|pinned compression|shear|uniaxial tension|uniaxial compression"), 
                          "Load case (pinned tension|pinned compression|shear|uniaxial tension|uniaxial compression)");
                          
        prm.declare_entry("Load type", "neumann", Patterns::Selection("neumann|dirichlet"), 
                          "Load type (neumann|dirichlet)");
    }
    prm.leave_subsection();
}

void Geometry::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry");
    {
        shape = prm.get("Shape");
        elements_per_edge = prm.get_integer("Elements per edge");
        scale = prm.get_double("Grid scale");
        load_scale = prm.get_double("Load scale");
        load_case = prm.get("Load case");
        load_type = prm.get("Load type");
    }
    prm.leave_subsection();
}

struct Materials
{
    double nu;
    double mu;
    double E;
    double sigma_y;
    double H_bar_prime;
    double bt;
    std::string material_model;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Materials::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        prm.declare_entry("Poisson's ratio", "0.3", Patterns::Double(-1.0, 0.5), "Poisson's ratio");

        prm.declare_entry("Shear modulus", "4.6153846e+03", Patterns::Double(), "Shear modulus");

        prm.declare_entry("Young's modulus", "12000.0", Patterns::Double(), "Young's modulus");

        prm.declare_entry("Yield stress", "100.0", Patterns::Double(), "Yield stress");

        prm.declare_entry("Hardening parameter", "1003.0", Patterns::Double(), "Hardening parameter");

        prm.declare_entry("beta", "0.9970089730807577", Patterns::Double(), "beta");

        prm.declare_entry("Material model", "J2", Patterns::Selection("Neo Hookian Midterm|Option 0|Linear|Homework 2|J2"), 
                            "Material model (Neo Hookian Midterm|Option 0|Linear|Homework 2|J2)");
    }
    prm.leave_subsection();
}

void Materials::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        nu = prm.get_double("Poisson's ratio");
        mu = prm.get_double("Shear modulus");
        E  = prm.get_double("Young's modulus");
        sigma_y  = prm.get_double("Yield stress");
        H_bar_prime  = prm.get_double("Hardening parameter");
        bt  = prm.get_double("beta");
        material_model = prm.get("Material model");
    }
    prm.leave_subsection();
}
struct LinearSolver
{
    std::string type_lin;
    double tol_lin;
    double max_iterations_lin;
    std::string preconditioner_type;
    double preconditioner_relaxation;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void LinearSolver::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Linear solver");
    {
        prm.declare_entry("Solver type", "CG", Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual", "1e-6", Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry("Max iteration multiplier", "1", Patterns::Double(0.0),
                          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type", "ssor", Patterns::Selection("jacobi|ssor"), "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation", "0.65", Patterns::Double(0.0),
                          "Preconditioner relaxation value");
    }
    prm.leave_subsection();
}

void LinearSolver::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Linear solver");
    {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
        preconditioner_type = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
    }
    prm.leave_subsection();
}
struct NonlinearSolver
{
    unsigned int max_iterations_NR;
    double tol_f;
    double tol_u;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void NonlinearSolver::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Nonlinear solver");
    {
        prm.declare_entry("Max iterations Newton-Raphson", "10", Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force", "1.0e-9", Patterns::Double(0.0), "Force residual tolerance");

        prm.declare_entry("Tolerance displacement", "1.0e-6", Patterns::Double(0.0), "Displacement error tolerance");
    }
    prm.leave_subsection();
}

void NonlinearSolver::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Nonlinear solver");
    {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f = prm.get_double("Tolerance force");
        tol_u = prm.get_double("Tolerance displacement");
    }
    prm.leave_subsection();
}

struct Time
{
    double delta_t;
    double end_time;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Time::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time");
    {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time step size", "0.1", Patterns::Double(), "Time step size");
    }
    prm.leave_subsection();
}

void Time::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time");
    {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
    }
    prm.leave_subsection();
}

struct AllParameters : public AssemblyMethod,
                       public FESystem,
                       public Geometry,
                       public Materials,
                       public LinearSolver,
                       public NonlinearSolver,
                       public Time

{
    AllParameters(const std::string &input_file);

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

AllParameters::AllParameters(const std::string &input_file)
{
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(input_file);
    parse_parameters(prm);
}

void AllParameters::declare_parameters(ParameterHandler &prm)
{
    AssemblyMethod::declare_parameters(prm);
    FESystem::declare_parameters(prm);
    Geometry::declare_parameters(prm);
    Materials::declare_parameters(prm);
    LinearSolver::declare_parameters(prm);
    NonlinearSolver::declare_parameters(prm);
    Time::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm)
{
    AssemblyMethod::parse_parameters(prm);
    FESystem::parse_parameters(prm);
    Geometry::parse_parameters(prm);
    Materials::parse_parameters(prm);
    LinearSolver::parse_parameters(prm);
    NonlinearSolver::parse_parameters(prm);
    Time::parse_parameters(prm);
}
} // namespace Parameters

class Time
{
  public:
    Time(const double time_end, const double delta_t)
        : timestep(0), time_current(0.0), time_end(time_end), delta_t(delta_t)
    {
    }

    virtual ~Time()
    {
    }

    double current() const
    {
        return time_current;
    }
    double end() const
    {
        return time_end;
    }
    double get_delta_t() const
    {
        return delta_t;
    }
    unsigned int get_timestep() const
    {
        return timestep;
    }
    void increment()
    {
        time_current += delta_t;
        ++timestep;
    }

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
};

template <int dim, typename NumberType> class Material_hw2
{
  public:
    Material_hw2(const double mu, const double nu)
        : mu(mu), nu(nu)
    {
    }

    ~Material_hw2()
    {
    }

    SymmetricTensor<2, dim, NumberType>  get_sigma_hw2_(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return get_stress(eps);
    }

    SymmetricTensor<4, dim, NumberType>  get_stiffness_hw2_(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return get_stiffness(eps);
    }

  private:
    const double mu;
    const double nu;

    const double a = 40;
    const double b = -50;
    const double c = -30;

    SymmetricTensor<2, dim, NumberType>  get_stress(const SymmetricTensor<2, dim, NumberType> &eps) const
    {       
        SymmetricTensor<2, dim, NumberType> tmp;
        tmp  = 2 * a * eps;
        tmp += 3 * b * trace(eps) * trace(eps) * Physics::Elasticity::StandardTensors<dim>::I;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int m = 0; m < dim; ++m)
                    tmp[i][j] += 3 * c * eps[i][m] * eps[j][m];   
        return tmp;
    }

    SymmetricTensor<4, dim, NumberType>  get_stiffness(const SymmetricTensor<2, dim, NumberType> &eps) const
    {

        SymmetricTensor<4, dim, NumberType> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                            tmp[i][j][k][l] =
                                (((i == k) && (j == l) ? a + 1.5 * c * (eps[j][l] + eps[i][k])
                                                    : 0.0) +
                                 ((i == l) && (j == k) ? a + 1.5 * c * (eps[j][k] + eps[i][l])
                                                     : 0.0) +
                                 ((i == j) && (k == l) ? 6 * b * trace(eps) : 0.0));
        return tmp;
    }

};

template <int dim, typename NumberType> class Material_linear_elastic
{
  public:
    Material_linear_elastic(const double mu, const double nu)
        : mu(mu), nu(nu)
    {
    }

    ~Material_linear_elastic()
    {
    }

    SymmetricTensor<2, dim, NumberType>  get_sigma_linear_(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return get_stress(eps);
    }

    SymmetricTensor<4, dim, NumberType>  get_D_linear_(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return get_stiffness(eps);
    }

  private:
    const double mu;
    const double nu;

    SymmetricTensor<2, dim, NumberType>  get_stress(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return (40*trace(eps)*Physics::Elasticity::StandardTensors<dim>::I + 2.0*40.0*eps);
    }

    SymmetricTensor<4, dim, NumberType>  get_stiffness(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        SymmetricTensor<4, dim, NumberType> tmp;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                        tmp[i][j][k][l] = (((i == k) && (j == l) ? 40.0 : 0.0) + 
                                           ((i == l) && (j == k) ? 40.0 : 0.0) +
                                           ((i == j) && (k == l) ? 40.0 : 0.0));
        return tmp;
    }

};

template <int dim, typename NumberType> class Material_j2_plastic
{
  public:
    Material_j2_plastic(const double mu, const double nu, const double sigma_y, const double H_bar_prime, const double bt)
        : mu(mu)
        , nu(nu)
        , sigma_y(sigma_y)
        , H_bar_prime(H_bar_prime)
        , bt(bt)

        , kappa(2.0 * mu * (1+nu) / 3.0/(1.0 - 2.0 * nu))
        , lambda(2*mu*nu/(1-2*nu))
        , f_tr(0.0)
        , alpha(0.0)
        , d_gamma(0.0)
        , beta(SymmetricTensor<2, dim>())
        , xi(SymmetricTensor<2, dim>())
        , dev_sigma_trial(SymmetricTensor<2, dim>())
        , dev_eps(SymmetricTensor<2, dim>())
        , dev_eps_p(SymmetricTensor<2, dim>())
        , sigma_new(SymmetricTensor<2, dim>())
        , C_ep(SymmetricTensor<4, dim>())
    {
    }

    ~Material_j2_plastic()
    {
    }

    SymmetricTensor<2, dim, NumberType>  get_sigma_j2_(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        return get_stress(eps);
    }

    SymmetricTensor<4, dim, NumberType>  get_D_j2_(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        return get_stiffness(eps);
    }

  protected:
    const double mu;
    const double nu;
    const double sigma_y;
    const double H_bar_prime;
    const double bt;

    const double kappa;
    const double lambda;

    double f_tr;
    double alpha;
    double d_gamma;

    SymmetricTensor<2, dim> beta;
    SymmetricTensor<2, dim> xi;

    SymmetricTensor<2, dim> dev_sigma_trial;
    SymmetricTensor<2, dim> dev_eps;
    SymmetricTensor<2, dim> dev_eps_p;
    SymmetricTensor<2, dim> n;

    SymmetricTensor<2, dim> sigma_new;
    SymmetricTensor<4, dim> C_ep;

    bool  check_yield(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        dev_eps = eps - (1.0 / 3.0) * trace(eps) * Physics::Elasticity::StandardTensors<dim>::I;
        dev_sigma_trial = 2.0*mu*(dev_eps - dev_eps_p);
        //std::cout<<", dev_eps  = "<<dev_eps<<std::endl;
        //std::cout<<", mu  = "<<mu<<std::endl;

        xi = dev_sigma_trial - beta;
        //std::cout<<", xi  = "<<xi<<std::endl;
        f_tr = xi.norm()  -  (sqrt(2.0 / 3.0) * (sigma_y + bt*H_bar_prime*alpha) );
        //std::cout<<", f_tr  = "<<f_tr<<std::endl;
        return (f_tr > 0.0);
    }

    void  calculate_stress(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        bool plastic = check_yield(eps);

        if (!plastic)
        {
            sigma_new = (lambda*trace(eps)*Physics::Elasticity::StandardTensors<dim>::I + 2.0*mu*eps);
        }
        else
        {
            sigma_new = plastic_step(eps);
        }
        return;
    }

    SymmetricTensor<2, dim, NumberType>  plastic_step(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        n = xi / xi.norm();
        //std::cout<<", n  = "<<n<<std::endl;

        d_gamma = f_tr / (2.0 * mu * (1.0 + H_bar_prime / (3.0 * mu)));
        std::cout<<", d_gamma  = "<<d_gamma<<std::endl;

        alpha += sqrt(2.0/3.0) * d_gamma;
        beta += sqrt(2.0/3.0) * sqrt(2.0/3.0) * (1.0-bt) * H_bar_prime * d_gamma * n;

        dev_eps_p += d_gamma * n;

        //std::cout<<", kappa * trace(eps)  = "<<kappa * trace(eps)<<std::endl;
        //std::cout<<", kappa  = "<<kappa<<std::endl;
        //std::cout<<", eps  = "<<eps<<std::endl;
        //std::cout<<", dev_sigma_trial  = "<<dev_sigma_trial<<std::endl;
SymmetricTensor<2, dim, NumberType> sigma = kappa * trace(eps) * Physics::Elasticity::StandardTensors<dim>::I + dev_sigma_trial - 2.0 * mu * d_gamma * n;
//std::cout<<sigma<<std::endl;

        return sigma;
    }

    void  calculate_stiffness(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        bool plastic = check_yield(eps);
        SymmetricTensor<4, dim, NumberType> tmp;
        if (!plastic)
        {
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                            C_ep[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) + 
                                               ((i == l) && (j == k) ? mu : 0.0) +
                                               ((i == j) && (k == l) ? lambda : 0.0));
        }
        else
        {
            double theta = 1.0 - 2.0 * mu * d_gamma / xi.norm();
            double theta_bar = 1.0 / (1.0 + (H_bar_prime)/(3.0 * mu)) - ( 1.0 - theta);
            C_ep = kappa * Physics::Elasticity::StandardTensors<dim>::IxI + 2.0 * mu * theta * Physics::Elasticity::StandardTensors<dim>::dev_P - 2.0 * mu * theta_bar * outer_product(n, n);
std::cout<<C_ep<<std::endl;
        }
        return;
    }

    SymmetricTensor<2, dim, NumberType>  get_stress(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        calculate_stress(eps);
        return sigma_new;
    }

    SymmetricTensor<4, dim, NumberType>  get_stiffness(const SymmetricTensor<2, dim, NumberType> &eps)
    {
        calculate_stiffness(eps);
        return C_ep;
    }
};

template <int dim, typename NumberType> class Material_Compressible_Neo_Hook_One_Field
{
  public:
    Material_Compressible_Neo_Hook_One_Field(const double mu, const double nu)
        : kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))), c_1(mu / 2.0)
    {
        Assert(kappa > 0, ExcInternalError());
    }

    ~Material_Compressible_Neo_Hook_One_Field()
    {
    }
    NumberType get_Psi(const NumberType &det_F, const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    }
    SymmetricTensor<2, dim, NumberType> get_tau(const NumberType &det_F,
                                                const SymmetricTensor<2, dim, NumberType> &b_bar)
    {

        return get_tau_vol(det_F) + get_tau_iso(b_bar);
    }
    SymmetricTensor<4, dim, NumberType> get_Jc(const NumberType &det_F,
                                               const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return get_Jc_vol(det_F) + get_Jc_iso(b_bar);
    }

  private:
    const double kappa;
    const double c_1;

    NumberType get_Psi_vol(const NumberType &det_F) const
    {
        return (kappa / 4.0) * (det_F * det_F - 1.0 - 2.0 * std::log(det_F));
    }

    NumberType get_Psi_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return c_1 * (trace(b_bar) - dim);
    }

    NumberType get_dPsi_vol_dJ(const NumberType &det_F) const
    {
        return (kappa / 2.0) * (det_F - 1.0 / det_F);
    }

    SymmetricTensor<2, dim, NumberType> get_tau_vol(const NumberType &det_F) const
    {
        return NumberType(get_dPsi_vol_dJ(det_F) * det_F) * Physics::Elasticity::StandardTensors<dim>::I;
    }
    
    SymmetricTensor<2, dim, NumberType> get_tau_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar(b_bar);
    }
    
    SymmetricTensor<2, dim, NumberType> get_tau_bar(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return 2.0 * c_1 * b_bar;
    }
    
    NumberType get_d2Psi_vol_dJ2(const NumberType &det_F) const
    {
        return ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
    }
    
    SymmetricTensor<4, dim, NumberType> get_Jc_vol(const NumberType &det_F) const
    {

        return det_F * ((get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F)) *
                            Physics::Elasticity::StandardTensors<dim>::IxI -
                        (2.0 * get_dPsi_vol_dJ(det_F)) * Physics::Elasticity::StandardTensors<dim>::S);
    }
    
    SymmetricTensor<4, dim, NumberType> get_Jc_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        const SymmetricTensor<2, dim> tau_bar = get_tau_bar(b_bar);
        const SymmetricTensor<2, dim> tau_iso = get_tau_iso(b_bar);
        const SymmetricTensor<4, dim> tau_iso_x_I =
            outer_product(tau_iso, Physics::Elasticity::StandardTensors<dim>::I);
        const SymmetricTensor<4, dim> I_x_tau_iso =
            outer_product(Physics::Elasticity::StandardTensors<dim>::I, tau_iso);
        const SymmetricTensor<4, dim> c_bar = get_c_bar();

        return (2.0 / dim) * trace(tau_bar) * Physics::Elasticity::StandardTensors<dim>::dev_P -
               (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
               Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar *
                   Physics::Elasticity::StandardTensors<dim>::dev_P;
    }
    
    SymmetricTensor<4, dim, double> get_c_bar() const
    {
        return SymmetricTensor<4, dim>();
    }
};

template <int dim, typename NumberType> class Material_Neo_Hookian
{
  public:
    Material_Neo_Hookian(const double mu, const double nu)
        : mu(mu), lambda((2.0 * mu * nu) / (1.0 - 2.0 * nu))
    {
    }

    ~Material_Neo_Hookian()
    {
    }

    SymmetricTensor<2, dim, NumberType> get_S(const SymmetricTensor<2, dim, NumberType> &C_inv, 
                                              const NumberType &J) const
    {
        return S_PKII(C_inv, J);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma(const SymmetricTensor<2, dim, NumberType> &b,
                                                  const NumberType &J,
                                                  const NumberType &J_inv) const
    {
        return get_sigma_cauchy(b, J, J_inv);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma_S(const SymmetricTensor<2, dim, NumberType> &S,
                                                    const Tensor<2, dim, NumberType> &F,
                                                    const NumberType &J_inv) const
    {
        return get_sigma_cauchy_from_S(S, F, J_inv);
    }

    SymmetricTensor<4, dim, NumberType> get_C_IJLK(const Tensor<2, dim, NumberType> &F,
                                                   const SymmetricTensor<2, dim, NumberType> C_inv,
                                                   const NumberType &J) const
    {
        return get_CIJKL(F,C_inv,J);
    }

    SymmetricTensor<4, dim, NumberType> get_c(const Tensor<2, dim, NumberType> F,
                                              const SymmetricTensor<4, dim, NumberType> &C_IJLK,
                                              const NumberType &J_inv) const
    {
        return get_c_ijlk(F,C_IJLK,J_inv);
    }

  private: 

    const double mu;
    const double lambda;
    
    // Get the second Piola-Kirchhoff stress tensor S_ij
    SymmetricTensor<2, dim, NumberType> S_PKII(const SymmetricTensor<2, dim, NumberType> &C_inv, 
                                               const NumberType &J) const
    {
        return (mu     * (Physics::Elasticity::StandardTensors<dim>::I - C_inv) + 
                lambda * std::log(J) * C_inv);
    }

    // Get the Cauchy stress tensor sigma_ij
    SymmetricTensor<2, dim, NumberType> get_sigma_cauchy(const SymmetricTensor<2, dim, NumberType> &b,
                                                         const NumberType &J,
                                                         const NumberType &J_inv) const
    {
        return (mu     * J_inv * (b - Physics::Elasticity::StandardTensors<dim>::I) + 
                lambda * J_inv * std::log(J) * Physics::Elasticity::StandardTensors<dim>::I);
    }

    // Get the Cauchy stress tensor sigma_ij
    SymmetricTensor<2, dim, NumberType> get_sigma_cauchy_from_S(const SymmetricTensor<2, dim, NumberType> &S,
                                                                const Tensor<2, dim, NumberType> &F,
                                                                const NumberType &J_inv) const
    {
        return symmetrize( J_inv * F * S * transpose(F));
    }

    // Get the second elasticity tensor C_IJKL = A(2) = C^{SE}
    // dC_inv_dC() returns - 1/2 * (C_IK^-1 * C_JL^-1 + C_IL^-1 * C_JK^-1)
    SymmetricTensor<4, dim, NumberType> get_CIJKL(const Tensor<2, dim, NumberType> &F,
                                                  const SymmetricTensor<2, dim, NumberType> &C_inv,
                                                  const NumberType &J) const
    {
        SymmetricTensor<4, dim> tmp;
        const SymmetricTensor<4, dim> C_inv_x_C_inv =  outer_product(C_inv,C_inv);
        const SymmetricTensor<4, dim> dC_inv_dC     =  -2.0* Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);

/*         for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = 0; l < dim; ++l)
                tmp[i][j][k][l] = ( mu * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k] )+ 
                                    lambda * (C_inv[i][j] * C_inv[k][l]  - 
                                    std::log(J) * (C_inv[i][k] * C_inv[j][l] + C_inv[i][l] * C_inv[j][k] )));
        return tmp; 
 */
        return (mu * dC_inv_dC + lambda * (C_inv_x_C_inv - std::log(J) * dC_inv_dC));
    }

    // Get spatial tangent moduli c_ijkl = F_im F_jn F_kp F_lq C^SE_mpqn
    SymmetricTensor<4, dim, NumberType> get_c_ijlk(const Tensor<2, dim, NumberType> F,
                                                   const SymmetricTensor<4, dim, NumberType> &C_IJLK,
                                                   const NumberType &J_inv) const
    {
/*         SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = 0; l < dim; ++l)
                tmp[i][j][k][l] = (((i == k) && (j == l) ? mu * J_inv - std::log(1.0/J_inv): 0.0) +
                                   ((i == l) && (j == k) ? mu * J_inv - std::log(1.0/J_inv): 0.0) +
                                   ((i == j) && (k == l) ? lambda * J_inv : 0.0));
        return tmp; */
        SymmetricTensor<4, dim> tmp;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
              for (unsigned int l = 0; l < dim; ++l)
                tmp[i][j][k][l] = (J_inv * F[i][i] * F[j][j] * F[k][k] * F[l][l]* C_IJLK[i][j][k][l]);
        return tmp;
/*         return (J_inv * symmetrize(F * F) * symmetrize(F * F) * C_IJLK); */
    }
};

template <int dim, typename NumberType> class PointHistory
{
  public:
    PointHistory()
    {
    }

    virtual ~PointHistory()
    {
    }

    void setup_lqp(const Parameters::AllParameters &parameters)
    {
        linear_elastic.reset(new Material_linear_elastic<dim, NumberType>(parameters.mu, parameters.nu));
        j2_plastic.reset(new Material_j2_plastic<dim, NumberType>(parameters.mu, parameters.nu,parameters.sigma_y,parameters.H_bar_prime,parameters.bt));
        hw2.reset(new Material_hw2<dim, NumberType>(parameters.mu, parameters.nu));
        material.reset(new Material_Compressible_Neo_Hook_One_Field<dim, NumberType>(parameters.mu, parameters.nu));
        Neo_Hookian.reset(new Material_Neo_Hookian<dim, NumberType>(parameters.mu, parameters.nu));
    }

    SymmetricTensor<2, dim, NumberType> get_sigma_j2(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return j2_plastic->get_sigma_j2_(eps);
    }

    SymmetricTensor<4, dim, NumberType> get_D_j2(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return j2_plastic->get_D_j2_(eps);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma_linear(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return linear_elastic->get_sigma_linear_(eps);
    }

    SymmetricTensor<4, dim, NumberType> get_D_linear(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return linear_elastic->get_D_linear_(eps);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma_hw2(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return hw2->get_sigma_hw2_(eps);
    }

    SymmetricTensor<4, dim, NumberType> get_stiffness_hw2(const SymmetricTensor<2, dim, NumberType> &eps) const
    {
        return hw2->get_stiffness_hw2_(eps);
    }

    SymmetricTensor<2, dim, NumberType> get_S(const SymmetricTensor<2, dim, NumberType> &C_inv, 
                                              const NumberType &J) const
    {
        return Neo_Hookian->get_S(C_inv, J);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma(const SymmetricTensor<2, dim, NumberType> &b, 
                                                  const NumberType &J,
                                                  const NumberType &J_inv) const
    {
        return Neo_Hookian->get_sigma(b, J, J_inv);
    }

    SymmetricTensor<2, dim, NumberType> get_sigma_S(const SymmetricTensor<2, dim, NumberType> &S,
                                                    const Tensor<2, dim, NumberType> &F,
                                                    const NumberType &J_inv) const
    {
        return Neo_Hookian->get_sigma_S(S, F, J_inv);
    }

    SymmetricTensor<4, dim, NumberType> get_C_IJLK(const Tensor<2, dim, NumberType> &F,
                                                   const SymmetricTensor<2, dim, NumberType> C_inv,
                                                   const NumberType &J) const
    {
        return Neo_Hookian->get_C_IJLK(F, C_inv, J);
    }

    SymmetricTensor<4, dim, NumberType> get_c(const Tensor<2, dim, NumberType> F,
                                              const SymmetricTensor<4, dim, NumberType> &C_IJLK,
                                              const NumberType &J_inv) const
    {
        return Neo_Hookian->get_c(F, C_IJLK ,J_inv);
    }

    NumberType get_Psi(const NumberType &det_F, const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return material->get_Psi(det_F, b_bar);
    }

    SymmetricTensor<2, dim, NumberType> get_tau(const NumberType &det_F,
                                                const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return material->get_tau(det_F, b_bar);
    }

    SymmetricTensor<4, dim, NumberType> get_Jc(const NumberType &det_F,
                                               const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
        return material->get_Jc(det_F, b_bar);
    }

  private:
    
    std::shared_ptr<Material_linear_elastic<dim, NumberType>> linear_elastic;
    std::shared_ptr<Material_j2_plastic<dim, NumberType>> j2_plastic;
    std::shared_ptr<Material_hw2<dim, NumberType>> hw2;
    std::shared_ptr<Material_Neo_Hookian<dim, NumberType>> Neo_Hookian;
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, NumberType>> material;
    
    
};

template <int dim, typename NumberType> struct Assembler_Base;
template <int dim, typename NumberType> struct Assembler;

template <int dim> class StrainPostprocessor : public DataPostprocessorTensor<dim>
{
  public:
    StrainPostprocessor()
        : DataPostprocessorTensor<dim>("strain", update_values | update_gradients | update_quadrature_points)
    {
    }

    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                       std::vector<Vector<double>> &E) const override
    {
        AssertDimension(input_data.solution_gradients.size(), E.size());

        for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
            AssertDimension(E[p].size(), (Tensor<2, dim>::n_independent_components));
            for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                    E[p][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(d, e))] =
                        (input_data.solution_gradients[p][d][e] + input_data.solution_gradients[p][e][d] + input_data.solution_gradients[p][d][e]*input_data.solution_gradients[p][e][d]) / 2.0;
        }
    }
};

template <int dim, typename NumberType> class Solid
{
  public:
    Solid(const Parameters::AllParameters &parameters);

    virtual ~Solid();

    void run();

  private:
    void make_grid();

    void system_setup();
    void assemble_system(const BlockVector<double> &solution_delta);

    friend struct Assembler_Base<dim, NumberType>;
    friend struct Assembler<dim, NumberType>;

    void make_constraints(const int &it_nr);

    void setup_qph();

    void solve_nonlinear_timestep(BlockVector<double> &solution_delta);

    std::pair<unsigned int, double> solve_linear_system(BlockVector<double> &newton_update);

    BlockVector<double> get_total_solution(const BlockVector<double> &solution_delta) const;
    void extract_stress() const;
    void output_results() const;
    const Parameters::AllParameters &parameters;

    double vol_reference;
    double vol_current;

    Triangulation<dim> triangulation;
    Time time;
    TimerOutput timer;
    CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory<dim, NumberType>> quadrature_point_history;

    const unsigned int degree;
    const FESystem<dim> fe;
    DoFHandler<dim> dof_handler_ref;
    const unsigned int dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;
    static const unsigned int n_blocks = 1;
    static const unsigned int n_components = dim;
    static const unsigned int first_u_component = 0;

    enum
    {
        u_dof = 0
    };

    std::vector<types::global_dof_index> dofs_per_block;
    const QGauss<dim> qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int n_q_points;
    const unsigned int n_q_points_f;

    AffineConstraints<double> constraints;
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> tangent_matrix;
    BlockVector<double> system_rhs;
    BlockVector<double> solution_n;
    struct Errors
    {
        Errors() : norm(1.0), u(1.0)
        {
        }

        void reset()
        {
            norm = 1.0;
            u = 1.0;
        }
        void normalise(const Errors &rhs)
        {
            if (rhs.norm != 0.0)
                norm /= rhs.norm;
            if (rhs.u != 0.0)
                u /= rhs.u;
        }

        double norm, u;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update, error_update_0, error_update_norm;

    void get_error_residual(Errors &error_residual);

    void get_error_update(const BlockVector<double> &newton_update, Errors &error_update);

    static void print_conv_header();

    void print_conv_footer();

    void print_tip_displacement();
};

template <int dim, typename NumberType>
Solid<dim, NumberType>::Solid(const Parameters::AllParameters &parameters)
    : parameters(parameters), vol_reference(0.0), vol_current(0.0),
      triangulation(Triangulation<dim>::maximum_smoothing), time(parameters.end_time, parameters.delta_t),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times), degree(parameters.poly_degree),

      fe(FE_Q<dim>(parameters.poly_degree), dim), dof_handler_ref(triangulation), dofs_per_cell(fe.dofs_per_cell),
      u_fe(first_u_component), dofs_per_block(n_blocks), qf_cell(parameters.quad_order), qf_face(parameters.quad_order),
      n_q_points(qf_cell.size()), n_q_points_f(qf_face.size())
{
}

template <int dim>
class IncrementalBoundaryValues : public Function<dim>
{
public:
IncrementalBoundaryValues(const double present_time, const double present_timestep) : 
Function<dim>(dim), velocity(0.2), present_time(present_time), present_timestep(present_timestep)
{}

virtual void vector_value(const Point<dim> &p, Vector<double> &  values) const override
{
    AssertDimension(values.size(), dim);

    if (p[0] == 1.0)
    {
    values    = 0;
    values(0) = present_timestep * velocity;
    }
}

virtual void
vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &  value_list) const override
{
    const unsigned int n_points = points.size();

    AssertDimension(value_list.size(), n_points);

    for (unsigned int p = 0; p < n_points; ++p)
        IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
}

private:
const double velocity;
const double present_time;
const double present_timestep;
};

template <int dim, typename NumberType> Solid<dim, NumberType>::~Solid()
{
    dof_handler_ref.clear();
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::run()
{
    make_grid();
    system_setup();
    output_results();
    time.increment();
    BlockVector<double> solution_delta(dofs_per_block);
    while (time.current() <= time.end())
    {
        solution_delta = 0.0;

        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;
        output_results();
        time.increment();
    }
    print_tip_displacement();
}

template <int dim> Point<dim> grid_y_transform(const Point<dim> &pt_in)
{
    const double &x = pt_in[0];
    const double &y = pt_in[1];

    const double y_upper = 1.0 + (2.0 / 1.0) * x;
    const double y_lower = 1.0 + (1.0 / 1.0) * x;
    const double theta = y / 2.0;
    const double y_transform = (1 - theta) * y_lower + theta * y_upper;

    Point<dim> pt_out = pt_in;
    pt_out[1] = y_transform;

    return pt_out;
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::make_grid()
{

    std::vector<unsigned int> repetitions(dim, parameters.elements_per_edge);

    if (dim == 3)
        repetitions[dim - 1] = 1;

    const Point<dim> bottom_left = (dim == 3 ? Point<dim>(0.0, 0.0, -0.5) : Point<dim>(0.0, 0.0));
    const Point<dim> top_right   = (dim == 3 ? Point<dim>(1.0, 1.0,  0.5) : Point<dim>(1.0, 1.0));

    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, bottom_left, top_right);
    const double tol_boundary = 1e-6;
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true)
            {
                if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
                    cell->face(face)->set_boundary_id(1);
                else if (std::abs(cell->face(face)->center()[0] - 1.0) < tol_boundary)
                    cell->face(face)->set_boundary_id(11);
                else if (std::abs(cell->face(face)->center()[1] - 0.0) < tol_boundary)
                    cell->face(face)->set_boundary_id(21);
                else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 0.5) < tol_boundary)
                    cell->face(face)->set_boundary_id(3);
            }

    if (parameters.shape == "transformed")
        GridTools::transform(&grid_y_transform<dim>, triangulation);

    GridTools::scale(parameters.scale, triangulation);

    vol_reference = GridTools::volume(triangulation);
    vol_current = vol_reference;
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::system_setup()
{
    timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(n_components, u_dof);
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: " << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs() << std::endl;

    tangent_matrix.clear();
    {
        const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];

        BlockDynamicSparsityPattern csp(n_blocks, n_blocks);

        csp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
        csp.collect_sizes();
        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
        for (unsigned int ii = 0; ii < n_components; ++ii)
            for (unsigned int jj = 0; jj < n_components; ++jj)
                coupling[ii][jj] = DoFTools::always;
        DoFTools::make_sparsity_pattern(dof_handler_ref, coupling, csp, constraints, false);
        sparsity_pattern.copy_from(csp);
    }

    tangent_matrix.reinit(sparsity_pattern);

    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    solution_n.reinit(dofs_per_block);
    solution_n.collect_sizes();
    setup_qph();

    timer.leave_subsection();
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::setup_qph()
{
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadrature_point_history.initialize(triangulation.begin_active(), triangulation.end(), n_q_points);

    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
    {
        const std::vector<std::shared_ptr<PointHistory<dim, NumberType>>> lqph =
            quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            lqph[q_point]->setup_lqp(parameters);
    }
}

template <int dim, typename NumberType>
void Solid<dim, NumberType>::solve_nonlinear_timestep(BlockVector<double> &solution_delta)
{
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ " << time.current() << "s" << std::endl;

    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
    {
        std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        make_constraints(newton_iteration);
        assemble_system(solution_delta);

        get_error_residual(error_residual);

        if (newton_iteration == 0)
            error_residual_0 = error_residual;
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u &&
            error_residual_norm.u <= parameters.tol_f)
        {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();

            break;
        }

        const std::pair<unsigned int, double> lin_solver_output = solve_linear_system(newton_update);

        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
            error_update_0 = error_update;

        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        solution_delta += newton_update;

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7) << std::scientific
                  << lin_solver_output.first << "  " << lin_solver_output.second << "  " << error_residual_norm.norm
                  << "  " << error_residual_norm.u << "  "
                  << "  " << error_update_norm.norm << "  " << error_update_norm.u << "  " << std::endl;
    }
    AssertThrow(newton_iteration <= parameters.max_iterations_NR, ExcMessage("No convergence in nonlinear solver!"));
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::print_conv_header()
{
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
    std::cout << std::endl;

    std::cout << "    SOLVER STEP    "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     NU_NORM     "
              << " NU_U " << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
    std::cout << std::endl;
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::print_conv_footer()
{
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
    std::cout << std::endl;

    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
              << "v / V_0:\t" << vol_current << " / " << vol_reference << std::endl;
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::print_tip_displacement()
{
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
    std::cout << std::endl;
    const Point<dim> soln_pt =
        (dim == 3 ? Point<dim>(1 * parameters.scale, 1 * parameters.scale, 1 * parameters.scale)
                  : Point<dim>(1 * parameters.scale, 1 * parameters.scale));
    double vertical_tip_displacement = 0.0;
    double vertical_tip_displacement_check_x = 0.0;
    double vertical_tip_displacement_check_y = 0.0;
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(), endc = dof_handler_ref.end();
    for (; cell != endc; ++cell)
    {

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            if (cell->vertex(v).distance(soln_pt) < 1e-6)
            {

                vertical_tip_displacement = solution_n(cell->vertex_dof_index(v, u_dof));

                const MappingQ<dim> mapping(parameters.poly_degree);
                const Point<dim> qp_unit = mapping.transform_real_to_unit_cell(cell, soln_pt);
                const Quadrature<dim> soln_qrule(qp_unit);
                AssertThrow(soln_qrule.size() == 1, ExcInternalError());
                FEValues<dim> fe_values_soln(fe, soln_qrule, update_values);
                fe_values_soln.reinit(cell);

                std::vector<Tensor<1, dim>> soln_values(soln_qrule.size());
                fe_values_soln[u_fe].get_function_values(solution_n, soln_values);
                vertical_tip_displacement_check_x = soln_values[0][u_dof];
                vertical_tip_displacement_check_y = soln_values[0][u_dof+1];
                break;
            }
    }
    //AssertThrow(vertical_tip_displacement > 0.0, ExcMessage("Found no cell with point inside!"))

        std::cout
        << "Top left displacement:\t Check: x = " << vertical_tip_displacement_check_x 
        << std::endl
        << "                      \t Check: y = " << vertical_tip_displacement_check_y
        << std::endl;
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::get_error_residual(Errors &error_residual)
{
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
        if (!constraints.is_constrained(i))
            error_res(i) = system_rhs(i);

    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_dof).l2_norm();
}

template <int dim, typename NumberType>
void Solid<dim, NumberType>::get_error_update(const BlockVector<double> &newton_update, Errors &error_update)
{
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
        if (!constraints.is_constrained(i))
            error_ud(i) = newton_update(i);

    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_dof).l2_norm();
}

template <int dim, typename NumberType>
BlockVector<double> Solid<dim, NumberType>::get_total_solution(const BlockVector<double> &solution_delta) const
{
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
}

template <int dim, typename NumberType> struct Assembler_Base
{
    virtual ~Assembler_Base()
    {
    }
    struct PerTaskData_ASM
    {
        const Solid<dim, NumberType> *solid;
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_ASM(const Solid<dim, NumberType> *solid)
            : solid(solid), cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell), cell_rhs(solid->dofs_per_cell),
              local_dof_indices(solid->dofs_per_cell)
        {
        }

        void reset()
        {
            cell_matrix = 0.0;
            cell_rhs = 0.0;
        }
    };

    struct ScratchData_ASM
    {
        const BlockVector<double> &solution_total;
        std::vector<Tensor<2, dim, NumberType>> solution_grads_u_total;

        FEValues<dim> fe_values_ref;
        FEFaceValues<dim> fe_face_values_ref;

        std::vector<std::vector<Tensor<2, dim, NumberType>>> grad_Nx;
        std::vector<std::vector<SymmetricTensor<2, dim, NumberType>>> symm_grad_Nx;

        ScratchData_ASM(const FiniteElement<dim> &fe_cell, const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                        const QGauss<dim - 1> &qf_face, const UpdateFlags uf_face,
                        const BlockVector<double> &solution_total)
            : solution_total(solution_total), solution_grads_u_total(qf_cell.size()),
              fe_values_ref(fe_cell, qf_cell, uf_cell), fe_face_values_ref(fe_cell, qf_face, uf_face),
              grad_Nx(qf_cell.size(), std::vector<Tensor<2, dim, NumberType>>(fe_cell.dofs_per_cell)),
              symm_grad_Nx(qf_cell.size(), std::vector<SymmetricTensor<2, dim, NumberType>>(fe_cell.dofs_per_cell))
        {
        }

        ScratchData_ASM(const ScratchData_ASM &rhs)
            : solution_total(rhs.solution_total), solution_grads_u_total(rhs.solution_grads_u_total),
              fe_values_ref(rhs.fe_values_ref.get_fe(), rhs.fe_values_ref.get_quadrature(),
                            rhs.fe_values_ref.get_update_flags()),
              fe_face_values_ref(rhs.fe_face_values_ref.get_fe(), rhs.fe_face_values_ref.get_quadrature(),
                                 rhs.fe_face_values_ref.get_update_flags()),
              grad_Nx(rhs.grad_Nx), symm_grad_Nx(rhs.symm_grad_Nx)
        {
        }

        void reset()
        {
            const unsigned int n_q_points = fe_values_ref.get_quadrature().size();
            const unsigned int n_dofs_per_cell = fe_values_ref.dofs_per_cell;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
                Assert(grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
                Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

                solution_grads_u_total[q_point] = Tensor<2, dim, NumberType>();
                for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
                {
                    grad_Nx[q_point][k] = Tensor<2, dim, NumberType>();
                    symm_grad_Nx[q_point][k] = SymmetricTensor<2, dim, NumberType>();
                }
            }
        }
    };
    void assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell, ScratchData_ASM &scratch,
                                  PerTaskData_ASM &data)
    {
        const Parameters::AllParameters &parameters = data.solid->parameters;
        // Neo Hookian Midterm|Option 0|Linear|Homework 2
        if (parameters.material_model == "Neo Hookian Midterm")
            assemble_system_tangent_neo_hookian(cell, scratch, data);
        else if (parameters.material_model == "Option 0")
            assemble_system_tangent_option_0(cell, scratch, data);
        else if (parameters.material_model == "Linear")
            assemble_system_tangent_linear_elastic(cell, scratch, data);
        else if (parameters.material_model == "Homework 2")
            assemble_system_tangent_hw2(cell, scratch, data);
        else if (parameters.material_model == "J2")
            assemble_system_tangent_J2(cell, scratch, data);
        else 
            AssertThrow(false, ExcMessage("Unknown material model"));
        
        assemble_rhs(cell, scratch, data);
    }

    void copy_local_to_global_ASM(const PerTaskData_ASM &data)
    {
        const AffineConstraints<double> &constraints = data.solid->constraints;
        BlockSparseMatrix<double> &tangent_matrix = const_cast<Solid<dim, NumberType> *>(data.solid)->tangent_matrix;
        BlockVector<double> &system_rhs = const_cast<Solid<dim, NumberType> *>(data.solid)->system_rhs;

        constraints.distribute_local_to_global(data.cell_matrix, data.cell_rhs, data.local_dof_indices, tangent_matrix,
                                               system_rhs);
    }

  protected:
    virtual void assemble_system_tangent_option_0(const typename DoFHandler<dim>::active_cell_iterator &,
                                                  ScratchData_ASM &, PerTaskData_ASM &)
    {
        AssertThrow(false, ExcPureFunctionCalled());
    }

    virtual void assemble_system_tangent_neo_hookian(const typename DoFHandler<dim>::active_cell_iterator &,
                                                     ScratchData_ASM &, PerTaskData_ASM &)
    {
        AssertThrow(false, ExcPureFunctionCalled());
    }

    virtual void assemble_system_tangent_linear_elastic(const typename DoFHandler<dim>::active_cell_iterator &,
                                                ScratchData_ASM &, PerTaskData_ASM &)
    {
        AssertThrow(false, ExcPureFunctionCalled());
    }

    virtual void assemble_system_tangent_hw2(const typename DoFHandler<dim>::active_cell_iterator &,
                                             ScratchData_ASM &, PerTaskData_ASM &)
    {
        AssertThrow(false, ExcPureFunctionCalled());
    }

    virtual void assemble_system_tangent_J2(const typename DoFHandler<dim>::active_cell_iterator &,
                                             ScratchData_ASM &, PerTaskData_ASM &)
    {
        AssertThrow(false, ExcPureFunctionCalled());
    }

    void assemble_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {

        const unsigned int &n_q_points_f = data.solid->n_q_points_f;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const Parameters::AllParameters &parameters = data.solid->parameters;
        const Time &time = data.solid->time;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 11)
            {
                scratch.fe_face_values_ref.reinit(cell, face);

                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                {
                    const double time_ramp = (time.current() / time.end());
                    const double magnitude = (1.0 * parameters.load_scale) * time_ramp;
                    Tensor<1, dim> dir;

                    // 
                    if ((parameters.load_case == "pinned tension") || (parameters.load_case == "uniaxial tension"))
                        dir[0] = 1.0;  // x-direction
                    else if ((parameters.load_case == "uniaxial compression") || (parameters.load_case == "pinned compression"))
                        dir[0] = -1.0;  // x-direction
                    else if (parameters.load_case == "shear")
                        dir[1] = 1.0;  // y-direction 
                    else 
                        AssertThrow(false, ExcMessage("Unknown load case"));
                    
                    const Tensor<1, dim> traction = magnitude * dir;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int i_group = fe.system_to_base_index(i).first.first;

                        if (i_group == u_dof)
                        {
                            const unsigned int component_i = fe.system_to_component_index(i).first;
                            const double Ni = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                            const double JxW = scratch.fe_face_values_ref.JxW(f_q_point);

                            data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                        }
                    }
                }
            }
    }

    void assemble_rhs_shear(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {

        const unsigned int &n_q_points_f = data.solid->n_q_points_f;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const Parameters::AllParameters &parameters = data.solid->parameters;
        const Time &time = data.solid->time;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;

        if (parameters.load_type == "neumann")
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 11)
                {
                    scratch.fe_face_values_ref.reinit(cell, face);

                    for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                    {
                        const double time_ramp = (time.current() / time.end());
                        const double magnitude = (1.0 * parameters.load_scale) * time_ramp;
                        Tensor<1, dim> dir;
                        dir[1] = 1.0;  // y-direction
                        const Tensor<1, dim> traction = magnitude * dir;

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            const unsigned int i_group = fe.system_to_base_index(i).first.first;

/*                             if (i_group == u_dof)
                            { */
                                const unsigned int component_i = fe.system_to_component_index(i).first;
                                const double Ni = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                                const double JxW = scratch.fe_face_values_ref.JxW(f_q_point);

                                data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
/*                             } */
                        }
                    }
                }
            else
                for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                    if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 100)
                    {
                        scratch.fe_face_values_ref.reinit(cell, face);

                        for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                        {
                            const double time_ramp = (time.current() / time.end());
                            const double magnitude = (1.0 * parameters.load_scale) * time_ramp;
                            Tensor<1, dim> dir;
                            dir[1] = 1.0;  // y-direction
                            const Tensor<1, dim> traction = magnitude * dir;

                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                                if (i_group == u_dof)
                                {
                                    const unsigned int component_i = fe.system_to_component_index(i).first;
                                    const double Ni = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                                    const double JxW = scratch.fe_face_values_ref.JxW(f_q_point);

                                    data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                                }
                            }
                        }
                    }

    }
};

template <int dim> struct Assembler<dim, double> : Assembler_Base<dim, double>
{
    typedef double NumberType;
    using typename Assembler_Base<dim, NumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim, NumberType>::PerTaskData_ASM;

    virtual ~Assembler()
    {
    }

    virtual void assemble_system_tangent_option_0(const typename DoFHandler<dim>::active_cell_iterator &cell, 
                                                  ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {
        const unsigned int &n_q_points = data.solid->n_q_points;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;
        const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>> lqph =
            const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u_total);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];
            const Tensor<2, dim, NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
            const NumberType det_F = determinant(F);
            const Tensor<2, dim, NumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
            const SymmetricTensor<2, dim, NumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
            const Tensor<2, dim, NumberType> F_inv = invert(F);
            Assert(det_F > NumberType(0.0), ExcInternalError());

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_dof)
                {
                    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                    scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else
                    Assert(k_group <= u_dof, ExcInternalError());
            }

            const SymmetricTensor<2, dim, NumberType> tau = lqph[q_point]->get_tau(det_F, b_bar);
            const SymmetricTensor<4, dim, NumberType> Jc = lqph[q_point]->get_Jc(det_F, b_bar);
            const Tensor<2, dim, NumberType> tau_ns(tau);
            const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_dof)
                    data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
                else
                    Assert(i_group <= u_dof, ExcInternalError());

                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group = fe.system_to_base_index(j).first.first;

                    if ((i_group == j_group) && (i_group == u_dof))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc * symm_grad_Nx[j] * JxW;
                        if (component_i == component_j)
                            data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns * grad_Nx[j][component_j] * JxW;
                    }
                    else
                        Assert((i_group <= u_dof) && (j_group <= u_dof), ExcInternalError());
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }

    virtual void assemble_system_tangent_neo_hookian(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {

        const unsigned int &n_q_points = data.solid->n_q_points;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;
        const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Pointer to local quadrature point history
        const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>> lqph =
            const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        // Nabla(u)
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u_total);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];

            // Deformation gradient, its determinant and inverse
            const Tensor<2, dim, NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
            const NumberType det_F = determinant(F);
            const Tensor<2, dim, NumberType> F_inv = invert(F);

            Assert(det_F > NumberType(0.0), ExcInternalError());
            
            // Right Cauchy-Green deformation tensor C = transpose(F) * F
            // and it's inverse
            const SymmetricTensor<2, dim, NumberType> C_G = Physics::Elasticity::Kinematics::C(F);
            const SymmetricTensor<2, dim, NumberType> C_inv = invert(C_G);

            // Left Cauchy-Green deformation tensor b = F * transpose(F)
            const SymmetricTensor<2, dim, NumberType> b = Physics::Elasticity::Kinematics::b(F);

            // Green-Lagrange strain tensor
            const SymmetricTensor<2, dim, NumberType> E = Physics::Elasticity::Kinematics::E(F);

            // Volume ratio and its inverse
            const NumberType J = det_F;
            const NumberType J_inv = NumberType(1.0) / J;


            SymmetricTensor<2, dim, NumberType> S = lqph[q_point]->get_S(C_inv, J);
            SymmetricTensor<2, dim, NumberType> sigma = lqph[q_point]->get_sigma_S(S, F, J_inv);
            //SymmetricTensor<2, dim, NumberType> sigma = lqph[q_point]->get_sigma(b, J ,J_inv);
            SymmetricTensor<4, dim, NumberType> C_IJLK = lqph[q_point]->get_C_IJLK(F, C_inv, J);
            SymmetricTensor<4, dim, NumberType> c = lqph[q_point]->get_c(F, C_IJLK, J_inv);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_dof)
                {
                    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F * J_inv;
                    scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else
                    Assert(k_group <= u_dof, ExcInternalError());
            }

            const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_dof)
                    data.cell_rhs(i) -= symm_grad_Nx[i] * sigma * JxW;
                else
                    Assert(i_group <= u_dof, ExcInternalError());

                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group = fe.system_to_base_index(j).first.first;

                    if ((i_group == j_group) && (i_group == u_dof))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i] * c * symm_grad_Nx[j] * JxW;
                        
                        data.cell_matrix(i, j) += grad_Nx[i][component_i] * sigma * grad_Nx[j][component_j] * JxW;
                    }
                    else
                        Assert((i_group <= u_dof) && (j_group <= u_dof), ExcInternalError());
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }

    virtual void assemble_system_tangent_linear_elastic(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {
        const unsigned int &n_q_points = data.solid->n_q_points;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;
        const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Pointer to local quadrature point history
        const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>> lqph =
            const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        // Nabla(u)
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u_total);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_dof)
                {
                    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point);
                    scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else
                    Assert(k_group <= u_dof, ExcInternalError());
            }

            const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);
            const SymmetricTensor<2, dim> eps = symmetrize(grad_u);

            const SymmetricTensor<2, dim> sigma = lqph[q_point]->get_sigma_linear(eps);
            const SymmetricTensor<4, dim> D = lqph[q_point]->get_D_linear(eps);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_dof)
                    
                    data.cell_rhs(i) -= symm_grad_Nx[i] * sigma * JxW;
                else
                    Assert(i_group <= u_dof, ExcInternalError());

                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group = fe.system_to_base_index(j).first.first;

                    if ((i_group == j_group) && (i_group == u_dof))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i]  * D * symm_grad_Nx[j] * JxW;
                    }
                    else
                        Assert((i_group <= u_dof) && (j_group <= u_dof), ExcInternalError());
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }

    virtual void assemble_system_tangent_hw2(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {
        const unsigned int &n_q_points = data.solid->n_q_points;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;
        const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Pointer to local quadrature point history
        const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>> lqph =
            const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        // Nabla(u)
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u_total);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_dof)
                {
                    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point);
                    scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else
                    Assert(k_group <= u_dof, ExcInternalError());
            }

            const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);
            const SymmetricTensor<2, dim> eps = symmetrize(grad_u);

            const SymmetricTensor<2, dim> sigma = lqph[q_point]->get_sigma_hw2(eps);
            const SymmetricTensor<4, dim> D = lqph[q_point]->get_stiffness_hw2(eps);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_dof)
                    
                    data.cell_rhs(i) -= symm_grad_Nx[i] * sigma * JxW;
                else
                    Assert(i_group <= u_dof, ExcInternalError());

                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group = fe.system_to_base_index(j).first.first;

                    if ((i_group == j_group) && (i_group == u_dof))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i]  * D * symm_grad_Nx[j] * JxW;
                    }
                    else
                        Assert((i_group <= u_dof) && (j_group <= u_dof), ExcInternalError());
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }

    virtual void assemble_system_tangent_J2(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData_ASM &scratch, PerTaskData_ASM &data)
    {
        const unsigned int &n_q_points = data.solid->n_q_points;
        const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
        const FESystem<dim> &fe = data.solid->fe;
        const unsigned int &u_dof = data.solid->u_dof;
        const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Pointer to local quadrature point history
        const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>> lqph =
            const_cast<const Solid<dim, NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        // Nabla(u)
        scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total, scratch.solution_grads_u_total);
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            const Tensor<2, dim, NumberType> &grad_u = scratch.solution_grads_u_total[q_point];

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;

                if (k_group == u_dof)
                {
                    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point);
                    scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
                else
                    Assert(k_group <= u_dof, ExcInternalError());
            }

            const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
            const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
            const double JxW = scratch.fe_values_ref.JxW(q_point);
            const SymmetricTensor<2, dim> eps = symmetrize(grad_u);

            const SymmetricTensor<2, dim> sigma = lqph[q_point]->get_sigma_j2(eps);
            const SymmetricTensor<4, dim> D = lqph[q_point]->get_D_j2(eps);

            // std::cout << "sigma = " << sigma << std::endl;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_dof)
                    
                    data.cell_rhs(i) -= symm_grad_Nx[i] * sigma * JxW;
                else
                    Assert(i_group <= u_dof, ExcInternalError());

                for (unsigned int j = 0; j <= i; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const unsigned int j_group = fe.system_to_base_index(j).first.first;

                    if ((i_group == j_group) && (i_group == u_dof))
                    {
                        data.cell_matrix(i, j) += symm_grad_Nx[i]  * D * symm_grad_Nx[j] * JxW;
                    }
                    else
                        Assert((i_group <= u_dof) && (j_group <= u_dof), ExcInternalError());
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }
};

template <int dim, typename NumberType>
void Solid<dim, NumberType>::assemble_system(const BlockVector<double> &solution_delta)
{
    timer.enter_subsection("Assemble linear system");
    std::cout << " ASM " << std::flush;

    tangent_matrix = 0.0;
    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_gradients | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_JxW_values);

    const BlockVector<double> solution_total(get_total_solution(solution_delta));
    typename Assembler_Base<dim, NumberType>::PerTaskData_ASM per_task_data(this);
    typename Assembler_Base<dim, NumberType>::ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face,
                                                                           solution_total);
    Assembler<dim, NumberType> assembler;

    WorkStream::run(dof_handler_ref.begin_active(), dof_handler_ref.end(),
                    static_cast<Assembler_Base<dim, NumberType> &>(assembler),
                    &Assembler_Base<dim, NumberType>::assemble_system_one_cell,
                    &Assembler_Base<dim, NumberType>::copy_local_to_global_ASM, scratch_data, per_task_data);

    timer.leave_subsection();
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::make_constraints(const int &it_nr)
{
    std::cout << " CST " << std::flush;

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    if (it_nr > 1)
        return;
    const bool apply_dirichlet_bc = (it_nr == 0);
    if (apply_dirichlet_bc)
    {
        constraints.clear();

        if ((parameters.load_case == "pinned tension") || (parameters.load_case == "pinned compression"))
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints, fe.component_mask(x_displacement));
        }
        if ((parameters.load_case == "pinned tension") || (parameters.load_case == "pinned compression"))
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints, fe.component_mask(y_displacement));
        }
        if (parameters.load_type == "dirichlet")
        {
            const int boundary_id = 11;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     IncrementalBoundaryValues<dim>(time.current() , time.current() ), constraints,
                                                     fe.component_mask(u_fe));
        }
        if ((parameters.load_case == "uniaxial tension") || (parameters.load_case == "uniaxial compression"))
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints, fe.component_mask(x_displacement));
        }
        if ((parameters.load_case == "uniaxial tension") || (parameters.load_case == "uniaxial compression"))
        {
            const int boundary_id = 21;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints, fe.component_mask(y_displacement));
        }
        if (parameters.load_case == "shear")
        {
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints);
        }
        if (dim == 3)
        {
            const int boundary_id = 3;
            const FEValuesExtractors::Scalar z_displacement(2);
            VectorTools::interpolate_boundary_values(dof_handler_ref, boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), constraints,
                                                     fe.component_mask(z_displacement));
        }
    }
    else
    {
        if (constraints.has_inhomogeneities())
        {
            AffineConstraints<double> homogeneous_constraints(constraints);
            for (unsigned int dof = 0; dof != dof_handler_ref.n_dofs(); ++dof)
                if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                    homogeneous_constraints.set_inhomogeneity(dof, 0.0);
            constraints.clear();
            constraints.copy_from(homogeneous_constraints);
        }
    }

    constraints.close();
}

template <int dim, typename NumberType>
std::pair<unsigned int, double> Solid<dim, NumberType>::solve_linear_system(BlockVector<double> &newton_update)
{
    BlockVector<double> A(dofs_per_block);
    BlockVector<double> B(dofs_per_block);

    unsigned int lin_it = 0;
    double lin_res = 0.0;

    {
        timer.enter_subsection("Linear solver");
        std::cout << " SLV " << std::flush;
        if (parameters.type_lin == "CG")
        {
            const int solver_its =
                static_cast<unsigned int>(tangent_matrix.block(u_dof, u_dof).m() * parameters.max_iterations_lin);
            const double tol_sol = parameters.tol_lin * system_rhs.block(u_dof).l2_norm();

            SolverControl solver_control(solver_its, tol_sol);

            GrowingVectorMemory<Vector<double>> GVM;
            SolverCG<Vector<double>> solver_CG(solver_control, GVM);
            PreconditionSelector<SparseMatrix<double>, Vector<double>> preconditioner(
                parameters.preconditioner_type, parameters.preconditioner_relaxation);
            preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

            solver_CG.solve(tangent_matrix.block(u_dof, u_dof), newton_update.block(u_dof), system_rhs.block(u_dof),
                            preconditioner);

            lin_it = solver_control.last_step();
            lin_res = solver_control.last_value();
        }
        else if (parameters.type_lin == "Direct")
        {
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
            A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

            lin_it = 1;
            lin_res = 0.0;
        }
        else
            Assert(false, ExcMessage("Linear solver type not implemented"));

        timer.leave_subsection();
    }
    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
}

template <int dim, typename NumberType> void Solid<dim, NumberType>::output_results() const
{
    {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n, solution_name, DataOut<dim>::type_dof_data, data_component_interpretation);
    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
        soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);

    std::ostringstream filename;
    filename << "solution-" << time.get_timestep() << ".vtk";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
    }
    StrainPostprocessor<dim> strain_postprocessor;

    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n, solution_name, DataOut<dim>::type_dof_data, data_component_interpretation);
    data_out.add_data_vector(solution_n, strain_postprocessor);
    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
        soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);

    std::ostringstream filename;
    filename << "solution-" << time.get_timestep() << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);
}

} // namespace Finite_strain_Neo_Hookian

int main(int argc, char *argv[])
{
    using namespace dealii;
    using namespace Finite_strain_Neo_Hookian;

    const unsigned int dim = 3;
    try
    {
        deallog.depth_console(0);
        Parameters::AllParameters parameters("parameters.prm");
        if (parameters.automatic_differentiation_order == 0)
        {
            std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

            Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int);

            typedef double NumberType;
            Solid<dim, NumberType> solid_3d(parameters);
            solid_3d.run();
        }
        else
        {
            AssertThrow(false, ExcMessage("The selected assembly method is not supported. "
                                          "You need deal.II 9.0 and Trilinos with the Sacado package "
                                          "to enable assembly using automatic differentiation."));
        }
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