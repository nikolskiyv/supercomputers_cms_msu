#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <omp.h>

#define LOGGER_PRETTY_TIME_FORMAT "%Y-%m-%d %H:%M:%S"


using namespace std;


// const int M = 80, N = 90;
const int M = 160, N = 180;
int monte_carlo_iterations_num = 1000;

const double A1 = -2.0, B1 = 5.0, A2 = -2.0, B2 = 6.0;  // Точки, определяющие область П. 1 по X. 2 по Y.
const double h1 = (B1 - A1) / M, h2 = (B2 - A2) / N;  // Размеры ячеек решетки по осям. h1 по X. h2 по Y.
double delta = 0.000001;  // Критерий остановки расчета.
const double eps = max(h1, h2) * max(h2, h2);

const string output_file_format = ".txt";

class TimeLogger
{
    ostream & out;
    string const msg;
public:
    TimeLogger(ostream& o)
    : out(o)
    { }

    template <typename T>
    ostream& operator<<(T const & x)
    {

        auto tp = chrono::system_clock::now();
        time_t current_time = chrono::system_clock::to_time_t(tp);

        tm* time_info = localtime(&current_time);

        char buffer[128];

        int string_size = strftime(
            buffer, sizeof(buffer),
            LOGGER_PRETTY_TIME_FORMAT,
            time_info
        );

        return out << "[ " << string(buffer, buffer + string_size) << " ] " << x;
    }
};

TimeLogger TimeLog(cout);

struct Point {
    double X;
    double Y;
};

struct Rectangle {
    Point top_left;
    Point bottom_right;
};


// Точки, определяющие область D. Прямоугольный треугольник.
Point A = Point { 3, 0 };
Point B = Point { 0, 4 };
Point C = Point { 0, 0 };


// Область П
Rectangle R = Rectangle{ Point {A1, B2}, Point{B1, A2} };


// Принадлежит ли точка области D
bool is_point_in_area_D(const Point& P) {
    if ( P.X > A.X || P.X < C.X || P.Y > B.Y || P.Y < C.Y ) return false;
    if ( P.Y > (- 4.0 / 3.0) * P.X + 4.0 ) return false;  // Точка лежит ниже биссектрисы
 
    return true;
}


// Принадлежит ли точка области R
bool is_point_in_rectangle_R(const Point& P) {
    if ( P.X < R.top_left.X || P.X > R.bottom_right.X || P.Y > R.top_left.Y || P.Y < R.bottom_right.Y ) return false;

    return true;
}


void matrix_to_stream(ofstream& stream, vector<double> matrix) {
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) stream << matrix[i * (M + 1) + j] << " ";
        stream << endl;
    }
}


// Скалярное произведение векторов
double calc_scalar_multiplication(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    int i;

    #pragma omp parallel for private(i) shared(v1, v2) reduction(+: result)
    for (i = 0; i < v1.size(); i++) result += v1[i] * v2[i];

    return result * h1 * h2;
}

// Евклидова норма
double calc_euclidean_norm(const vector<double>& v) {
    return sqrt(calc_scalar_multiplication(v, v));
}


// Вычитание векторов
vector<double> subtract_vector_from_vector(const vector<double>& v1, const vector<double>& v2) {
    int i;
    int v1_size = v1.size();
    vector<double> result(v1_size);


    #pragma omp parallel  private(i) shared(v1, v2, result, v1_size)
    #pragma omp for nowait schedule(static)
    for (i = 0; i < v1_size; i++) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

vector<double> multiply_vector_on_const(const vector<double>& v, double value) {
    int i;
    int v_size = v.size();
    vector<double> result(v_size);

    #pragma omp parallel shared(v, value, v_size) private(i)
    #pragma omp for nowait schedule(static)
    for (i = 0; i < v_size; i++) {
        result[i] = v[i] * value;
    }

    return result;
}

vector<double> multiply_matrix_on_vector(const vector<vector<double>>& matrix, const vector<double>& v) {
    vector<double> result((M + 1) * (N + 1));
    int i;

    #pragma omp parallel for private(i) shared(result, matrix, v) schedule(static)
    for (i = M + 2; i < matrix.size() - M - 2; i++) {
        result[i] = matrix[i][i] * v[i] 
                  + matrix[i][i - 1] * v[i - 1]
                  + matrix[i][i + 1] * v[i + 1]
                  + matrix[i][i - M - 1] * v[i - M - 1]
                  + matrix[i][i + M + 1] * v[i + M + 1];
    }
    return result;
}


// Коэффициент a_ij разностного уравнения
double calc__a_ij__coef(const vector<vector<Point>>& points, const int i, const int j) {

    Point p1 = Point{ points[i][j].X - h2 * 0.5, points[i][j].Y - h2 * 0.5 };  // Левая нижняя точка ячейки
    Point p2 = Point{ points[i][j].X - h2 * 0.5, points[i][j].Y + h2 * 0.5 };  // Левая верхняя точка ячейки

    // Для точек вне области D 1/eps
    if ( p1.Y > B.Y || p2.Y < C.Y ) return 1 / eps;

    int number_of_points_in_area_D = 0;

    // Генерация случайного Y из ячейки
    std::random_device rd_y;
    std::mt19937 gen_y(rd_y());
    std::uniform_real_distribution<> y(p1.Y, p2.Y);

    #pragma omp parallel for reduction(+: number_of_points_in_area_D)
    for (int i = 0; i < monte_carlo_iterations_num; i++) {
        double random_neighbor_y = y(gen_y);  // Случайная точка Y в области ячейки
        if ( is_point_in_area_D(Point{p1.X, random_neighbor_y}) ) number_of_points_in_area_D++;
    }
    
    double l = h2 * number_of_points_in_area_D / monte_carlo_iterations_num;  // Длина той части отрезка [P___i_j, P___i__j_inc_1], которая принадлежит D
    return (l / h2) + ((1 - l / h2) / eps);
}


// Коэффициент b_ij разностного уравнения
double calc__b_ij__coef(const vector<vector<Point>>& points, const int i, const int j) {

    Point p1 = Point{ points[i][j].X - h1 * 0.5, points[i][j].Y - h1 * 0.5 };  // Левая нижняя точка ячейки
    Point p2 = Point{ points[i][j].X + h1 * 0.5, points[i][j].Y - h1 * 0.5 };  // Правая нижняя точка ячейки

    // Для точек вне области D 1/eps
    if ( p1.X > A.X || p2.X < C.X ) return 1 / eps;

    int number_of_points_in_area_D = 0;

    // Генерация случайного X из ячейки
    std::random_device rd_x;
    std::mt19937 gen_x(rd_x());
    std::uniform_real_distribution<> x(p1.X, p2.X);

    #pragma omp parallel for reduction(+: number_of_points_in_area_D)
    for (int i = 0; i < monte_carlo_iterations_num; i++) {
        double random_neighbor_x = x(gen_x);
        if ( is_point_in_area_D(Point{random_neighbor_x, p1.Y}) ) number_of_points_in_area_D++;
    }

    double l = h1 * number_of_points_in_area_D / monte_carlo_iterations_num;
    return (l / h1) + ((1 - l / h1) / eps);
}


// Считаем площадь фиктивной области методом монте-карло
double calc_fictitious_area_square(Rectangle R) {

    int number_of_points_in_area_D = 0;

    // Генерируем случайные точки в области П
    std::random_device rd_x, rd_y;
    std::mt19937 gen_x(rd_x()), gen_y(rd_y());
    std::uniform_real_distribution<> x(R.top_left.X, R.bottom_right.X);
    std::uniform_real_distribution<> y(R.bottom_right.Y, R.top_left.Y);

    #pragma omp parallel for reduction(+: number_of_points_in_area_D)
    for (int i = 0; i < monte_carlo_iterations_num; i++) {
        double x_t = x(gen_x);
        double y_t = y(gen_y);
        if ( is_point_in_area_D(Point{ x_t, y_t }) ) number_of_points_in_area_D++;
    }
    return h1 * h2 * number_of_points_in_area_D / monte_carlo_iterations_num;
}


/*
Решение СЛАУ. Приближенное решение разностной схемы.
Решаем итерационным методом скорейшего спуска.

|| w - w__k || --> 0, k --> +inf
*/
vector<double> resolve_SLE(const vector<vector<double>>& A, const vector<double>& B) {

    TimeLog << "SLE resolving: START" << endl;

    vector<double> w__k(B.size(), 0.0);  // k(-ая) итерация сеточной ф-ии
    vector<double> w__k_inc_1(B.size());  // k+1(-ая) итерация сеточной ф-ии
    vector<double> r__k(w__k_inc_1.size());  // невязка

    vector<double> tau__k_inc_1___mul___r_k(w__k_inc_1.size());  
    vector<double> A___mul___r_k(r__k.size());
    vector<double> w__k_inc_1___sub___w_k(w__k_inc_1.size());

    double A___mul___r_k_norm, tau__k_inc_1, err;
    int iter;

    for (iter = 0; 1 == 1; iter++) {
        // Вычисление невязки t[k+1]
        r__k = multiply_matrix_on_vector(A, w__k);
        r__k = subtract_vector_from_vector(r__k, B);

        // Вычисление итерационного параметра
        A___mul___r_k = multiply_matrix_on_vector(A, r__k);
        A___mul___r_k_norm = calc_euclidean_norm(A___mul___r_k);
        tau__k_inc_1 = calc_scalar_multiplication(A___mul___r_k, r__k) / (A___mul___r_k_norm * A___mul___r_k_norm);

        // Вычисление k+1(-ой) итерации
        tau__k_inc_1___mul___r_k = multiply_vector_on_const(r__k, tau__k_inc_1);
        w__k_inc_1 = subtract_vector_from_vector(w__k, tau__k_inc_1___mul___r_k);

        // Разность k+1(-ой) и k(-ой) итераций. "Ошибка"
        w__k_inc_1___sub___w_k = subtract_vector_from_vector(w__k_inc_1, w__k);

        // Условие остановки
        err = calc_euclidean_norm(w__k_inc_1___sub___w_k);

        if ( iter % 2000 == 0 ) {
            TimeLog << "SLE resolving: err = " << err << endl;
            ofstream file( to_string(iter).append(output_file_format) );
            ofstream err_file( "err_dynamics.txt" );

            matrix_to_stream(file, w__k_inc_1___sub___w_k);
            err_file << err << " ";
        }

        if ( err < delta ) {
            ofstream file( to_string(iter).append("final").append(output_file_format) );
            matrix_to_stream(file, w__k_inc_1___sub___w_k);
            ofstream err_file( "err_dynamics.txt" );

            err_file << err << " ";
            break;
        }

        // Смещение
        w__k = w__k_inc_1;

    };

    TimeLog << "SUCCESSFULLY resolved SLE with <err = " << err << ", iterations_num = " << iter << ">" << endl;

    return w__k_inc_1;
}

void init_grid_points(vector<vector<Point>>& grid_points) {
    TimeLog << "grid points init: START" << endl;

    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            grid_points[i][j] = Point{ A1 + j * h1, A2 + i * h2 };
        }
    }

    TimeLog << "grid points init: SUCCESS" << endl;

};

void init_demicentral_nodes_squares_map(vector<double>& SQUARES_MAP, const vector<vector<Point>>& grid_points) {
    TimeLog << "demicentral nodes squares map init: START" << endl;

    int i, j;


    #pragma omp parallel for private(i, j) shared(SQUARES_MAP, grid_points) schedule(static)
    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {

            Point top_left = Point{ grid_points[i][j].X - h1 / 2, grid_points[i][j].Y + h2 / 2 };
            Point bottom_right = Point{ grid_points[i][j].X + h1 / 2, grid_points[i][j].Y - h2 / 2 };
            Rectangle rect = Rectangle{ top_left, bottom_right };

            SQUARES_MAP[i * (M + 1) + j] = calc_fictitious_area_square(rect) / (h1 * h2);
        }
    }

    TimeLog << "demicentral nodes squares map init: SUCCESS" << endl;

};

void init_A_operator(vector<vector<double>>& A, const vector<vector<Point>>& grid_points) {
    TimeLog << "init A operator: START" << endl;

    int i, j;
    double a___i__j, b___i__j, a___i__inc_1__j, b___i__j_inc_i;

    #pragma omp parallel for private(i, j, a___i__j, b___i__j, a___i__inc_1__j, b___i__j_inc_i) shared(A, grid_points)
    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {
            a___i__j = calc__a_ij__coef(grid_points, i, j);
            b___i__j = calc__b_ij__coef(grid_points, i, j);

            a___i__inc_1__j = calc__a_ij__coef(grid_points, i + 1, j);
            b___i__j_inc_i = calc__b_ij__coef(grid_points, i, j + 1);

            A[i * (M + 1) + j][i * (M + 1) + j] = (a___i__inc_1__j + a___i__j) / (h1 * h1) + (b___i__j_inc_i + b___i__j) / (h2 * h2);
            if ( i == 1 ) {
                A[i * (M + 1) + j][(i + 1) * (M + 1) + j] = - a___i__inc_1__j / (h1 * h1);
                if (j == 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
                else if (j == N - 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                }
                else {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
            }
            else if ( i == M - 1 ) {
                A[i * (M + 1) + j][(i - 1) * (M + 1) + j] = - a___i__j / (h1 * h1);;
                if (j == 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
                else if (j == N - 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                }
                else {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
            }
            else {
                A[i * (M + 1) + j][(i - 1) * (M + 1) + j] = - a___i__j / (h1 * h1);
                A[i * (M + 1) + j][(i + 1) * (M + 1) + j] = - a___i__inc_1__j / (h1 * h1);

                if (j == 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
                else if (j == N - 1) {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                }
                else {
                    A[i * (M + 1) + j][i * (M + 1) + j - 1] = - b___i__j / (h2 * h2);
                    A[i * (M + 1) + j][i * (M + 1) + j + 1] = - b___i__j_inc_i / (h2 * h2);
                }
            }
        }
    }

    TimeLog << "init A operator: SUCCESS" << endl;
}


int main() {
    cout << "CONFIG: "
         << "M = " << M
         << ", N = " << N
         << ", monte_carlo_iteratinos_num = " << monte_carlo_iterations_num 
         << ", h1 = " << h1
         << ", h2 = " << h2
         << ", delta = " << delta
         << ", eps = " << eps
         << endl;

    TimeLog << "START" << endl;

    vector<vector<Point>> grid_points(M + 1, vector<Point>(N + 1, Point{ 0.0, 0.0 }));
    vector<double> SQUARES_MAP((M + 1) * (N + 1), 0.0);
    vector<vector<double>> A((M + 1) * (N + 1), vector<double>((M + 1) * (N + 1), 0.0));

    auto start_ts = std::chrono::system_clock::now();

    init_grid_points(grid_points);
    init_demicentral_nodes_squares_map(SQUARES_MAP, grid_points);
    init_A_operator(A, grid_points);
    vector<double> result = resolve_SLE(A, SQUARES_MAP);

    auto end_ts = std::chrono::system_clock::now();

    TimeLog <<"Execution_time: "<< chrono::duration_cast<std::chrono::milliseconds>(end_ts - start_ts).count() / 1000.0 << "s" << std::endl;
    TimeLog << "SUCCESS" << endl;

    return 0;
}
