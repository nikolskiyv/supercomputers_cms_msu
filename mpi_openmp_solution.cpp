#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include <mpi.h>
#include <omp.h>


#define LOGGER_PRETTY_TIME_FORMAT "%Y-%m-%d %H:%M:%S"


using namespace std;


const int M = 80, N = 90;
// const int M = 160, N = 180;
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
        int rank, size;
    
        MPI_Comm_size(MPI_COMM_WORLD, &size);
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

void define_MPI_POINT(MPI_Datatype *tstype) {
    const int    count = 2;
    int          blocklens[count] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint     disps[count] = {offsetof(Point, X), offsetof(Point, Y)};

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}

struct Rectangle {
    Point top_left;
    Point bottom_right;
};

void define_MPI_RECTANGLE(MPI_Datatype *tstype) {
    MPI_Datatype MPI_POINT;
    define_MPI_POINT(&MPI_POINT);

    const int    count = 2;
    int          blocklens[count] = {1, 1};
    MPI_Datatype types[2] = {MPI_POINT, MPI_POINT};
    MPI_Aint     disps[count] = {offsetof(Rectangle, top_left), offsetof(Rectangle, bottom_right)};

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}



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


double calc_scalar_multiplication(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    int i;

    #pragma omp parallel for private(i) shared(v1, v2) reduction(+: result)
    for (i = 0; i < v1.size(); i++) {
        double tmp = v1[i] * v2[i];
        if (std::isnan(tmp)) {
            result += 0;
        } else {
            result += tmp;
        }
    }

    return result * h1 * h2;
}

// Евклидова норма
double local_calc_euclidean_norm(const vector<double>& v) {
    int size, rank;
    double result;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double local_norm = calc_scalar_multiplication(v, v);

    MPI_Reduce(&local_norm, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        result = sqrt(result);
    }
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return result;
}


// Вычитание векторов
vector<double> local_subtract_vector_from_vector(const vector<double>& v1, const vector<double>& v2) {
    int i;
    int v1_size = v1.size();
    vector<double> result(v1_size);

    #pragma omp parallel  private(i) shared(v1, v2, result, v1_size)
    #pragma omp for nowait schedule(static)
    for (i = 0; i < v1.size(); i++) {
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


vector<double> local_multiply_matrix_on_vector(double matrix[(M + 1) * (N + 1)][(M + 1) * (N + 1)], const vector<double>& v) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int items_per_worker = ((M + 1) * (N + 1)) / (size - 1);
    int tail = ((M + 1) * (N + 1)) - (size - 1) * items_per_worker;

    if (rank == size - 1) items_per_worker = tail;

    vector<int> displs(size, 0);
    vector<int> scounts(size, 0);

    for (int i = 0; i < size; i++) {
        scounts[i] = items_per_worker;
        displs[i] = (i == 0) ? 0 : displs[i-1] + items_per_worker;
    }

    int local_begin = displs[rank];

    vector<double> result((M + 1) * (N + 1));

    for (int local_idx = local_begin; local_idx < local_begin + items_per_worker; local_idx++) {
        if (local_idx < N + 2) {
            result[local_idx - local_begin] = 0.0;
        } else {
            result[local_idx - local_begin] = matrix[local_idx][local_idx] * v[local_idx] 
                                            + matrix[local_idx][local_idx - 1] * v[local_idx - 1]
                                            + matrix[local_idx][local_idx + 1] * v[local_idx + 1]
                                            + matrix[local_idx][local_idx - M - 1] * v[local_idx - M - 1]
                                            + matrix[local_idx][local_idx + M + 1] * v[local_idx + M + 1];
        }
    }
    return result;
}


// Коэффициент a_ij разностного уравнения
double calc__a_ij__coef(const vector<Point>& points, const int i, const int j) {
    Point p1 = Point{ points[i * N + j].X - h2 * 0.5, points[i * N + j].Y - h2 * 0.5 };  // Левая нижняя точка ячейки
    Point p2 = Point{ points[i * N + j].X - h2 * 0.5, points[i * N + j].Y + h2 * 0.5 };  // Левая верхняя точка ячейки

    // Для точек вне области D 1/eps
    if ( p1.Y > B.Y || p2.Y < C.Y ) return 1 / eps;

    int number_of_points_in_area_D = 0;

    // Генерация случайного Y из ячейки
    std::random_device rd_y;
    std::mt19937 gen_y(rd_y());
    std::uniform_real_distribution<> y(p1.Y, p2.Y);

    for (int i = 0; i < monte_carlo_iterations_num; i++) {
        double random_neighbor_y = y(gen_y);  // Случайная точка Y в области ячейки
        if ( is_point_in_area_D(Point{p1.X, random_neighbor_y}) ) number_of_points_in_area_D++;
    }
    
    double l = h2 * number_of_points_in_area_D / monte_carlo_iterations_num;  // Длина той части отрезка [P___i_j, P___i__j_inc_1], которая принадлежит D
    return (l / h2) + ((1 - l / h2) / eps);
}


// Коэффициент b_ij разностного уравнения
double calc__b_ij__coef(const vector<Point>& points, const int i, const int j) {
    Point p1 = Point{ points[i * N + j].X - h1 * 0.5, points[i * N + j].Y - h1 * 0.5 };  // Левая нижняя точка ячейки
    Point p2 = Point{ points[i * N + j].X + h1 * 0.5, points[i * N + j].Y - h1 * 0.5 };  // Правая нижняя точка ячейки

    // Для точек вне области D 1/eps
    if ( p1.X > A.X || p2.X < C.X ) return 1 / eps;

    int number_of_points_in_area_D = 0;

    // Генерация случайного X из ячейки
    std::random_device rd_x;
    std::mt19937 gen_x(rd_x());
    std::uniform_real_distribution<> x(p1.X, p2.X);

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

    for (int i = 0; i < monte_carlo_iterations_num; i++) {
        double x_t = x(gen_x);
        double y_t = y(gen_y);
        if ( is_point_in_area_D(Point{ x_t, y_t }) ) number_of_points_in_area_D++;
    }

    return h1 * h2 * number_of_points_in_area_D / monte_carlo_iterations_num;
}


void init_grid_points(vector<Point>& grid_points) {
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            grid_points[i * N + j] = Point{ A1 + i * h1, A2 + j * h2 };
        }
    }
};

// Параллельная инициализация по кускам строк
void async_init_demicentral_nodes_squares_map(vector<double>& SQUARES_MAP_local, const vector<Point>& grid_points, int local_row_nums, int offset) {
    // TimeLog << "demicentral nodes squares map init: START" << endl;
    int i, j;

    for (i = 1; i < local_row_nums; i++) {
        for (j = 1; j < N; j++) {
            Point top_left = Point{ grid_points[(i + local_row_nums * offset) * N + j].X - h1 / 2, grid_points[(i + local_row_nums * offset) * N + j].Y + h2 / 2 };
            Point bottom_right = Point{ grid_points[(i + local_row_nums * offset) * N + j].X + h1 / 2, grid_points[(i + local_row_nums * offset) * N + j].Y - h2 / 2 };
            Rectangle rect = Rectangle{ top_left, bottom_right };

            SQUARES_MAP_local[i * (N + 1) + j] = calc_fictitious_area_square(rect) / (h1 * h2);
        }
    }
};

int main(int argc, char** argv) {
    vector<Point> grid_points((N + 1) * (M + 1), Point{ 0.0, 0.0 });
    init_grid_points(grid_points);

    MPI_Init(&argc, &argv);

    int rank, size;
    
    int i;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double mpi_start_ts;
    if (rank == 0) {
        mpi_start_ts = MPI_Wtime();
    }

    auto start_ts = std::chrono::system_clock::now();

    /*  ==[ GLOBAL VARS ]==  */

    double A___mul___r_k_norm, tau__k_inc_1, local_tau__k_inc_1 = 0.0, err;

    vector<double> SQUARES_MAP_init((M + 1) * (N + 1), 0.0);
    vector<double> SQUARES_MAP((M + 1) * (N + 1));

    vector<double> w__k((M + 1) * (N + 1), 0.0);  // k(-ая) итерация сеточной ф-ии

    vector<double> w__k_inc_1((M + 1) * (N + 1));  // k+1(-ая) итерация сеточной ф-ии
    vector<double> r__k((M + 1) * (N + 1));  // невязка

    vector<double> tau__k_inc_1___mul___r_k((M + 1) * (N + 1));  
    vector<double> A___mul___r_k((M + 1) * (N + 1));
    vector<double> w__k_inc_1___sub___w_k((M + 1) * (N + 1));

    /*  ==[ MPI CUSTOM VARS ]==  */

    // Определяем кастомные типы Точка и Прямоугольник для пересылки
    MPI_Datatype MPI_POINT;
    define_MPI_POINT(&MPI_POINT);

    div_t div_t_rows_per_worker = div(M, size);  // Кол-во строк на воркер
    int rows_per_worker = div_t_rows_per_worker.quot;
    
    int items_per_worker = ((M + 1) * (N + 1)) / (size - 1);    
    int tail = ((M + 1) * (N + 1)) - (size - 1) * items_per_worker;

    /*  ==[ LOCAL WORKER VARS ]==  */

    vector<double> local_SQUARES_MAP(rows_per_worker * (N + 1));

    vector<double> local_w__k(items_per_worker);
    vector<double> local_w__k_inc_1(items_per_worker);
    vector<double> local_r__k(items_per_worker);

    vector<double> local_tau__k_inc_1___mul___r_k(items_per_worker);
    vector<double> local_A___mul___r_k(items_per_worker);
    vector<double> local_w__k_inc_1___sub___w_k(items_per_worker);

    /*  ==[ PARTS SIZE'S PER WORKER]==  */

    vector<int> displs_matrix(size, 0);
    vector<int> scounts_matrix(size, 0);

    for (int proc_idx = 0; proc_idx < size; proc_idx++) {
        int tail = (N + 1) % size;  // Остатки строк, тк не получается разбить между процессами равномерно

        displs_matrix[proc_idx] = proc_idx == size - 1 ? proc_idx * rows_per_worker * (N + 1) + tail : proc_idx * rows_per_worker * (N + 1);
        scounts_matrix[proc_idx] = proc_idx == size - 1 ? tail : rows_per_worker * (N + 1);
    }

    vector<int> displs(size, 0);
    vector<int> scounts(size, 0);

    for (int i = 0; i < size; i++) {
        scounts[i] = (i == size - 1) ? tail : items_per_worker;
        displs[i] = (i == 0) ? 0 : displs[i-1] + items_per_worker;
    }

    //  
    //  ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ МАТРИЦЫ ПЛОЩАДЕЙ
    //

    MPI_Scatterv(SQUARES_MAP_init.data(), scounts_matrix.data(), displs_matrix.data(), MPI_DOUBLE, local_SQUARES_MAP.data(), displs_matrix[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);  // SQUARES_MAP_init[rows_per_worker * N] -> proc[0, 1 ..]
    MPI_Bcast(grid_points.data(), (N + 1) * (M + 1), MPI_POINT, 0, MPI_COMM_WORLD);  // grid_points -> proc[0, 1 ..]
    MPI_Barrier(MPI_COMM_WORLD);  // Все процессы ждут

    // Распределяем инициализацию решетки по воркерам
    async_init_demicentral_nodes_squares_map(local_SQUARES_MAP, grid_points, rows_per_worker, rank);

    double A[(M+1)*(N+1)][(M+1)*(N+1)];
    double tmp;

    if (rank == 0) {
        // Синхронно определяем оператор
        int i, j;
        double a___i__j, b___i__j, a___i__inc_1__j, b___i__j_inc_i;

        TimeLog << "init A operator: START" << endl;

        for (i = 0; i < (M+1)*(N+1); i++) {
            for (j = 0; j < (M+1)*(N+1); j++) { // Инициализация массива A нулями
                A[i][j] = 0.0;
            }
        }

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

    MPI_Bcast(A, (M+1)*(N+1)*(M+1)*(N+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int iter;

    for (iter = 0; 1 == 1; iter++) {
        double tmp_v;

        local_r__k = local_multiply_matrix_on_vector(A, w__k);
        local_r__k = local_subtract_vector_from_vector(local_r__k, local_SQUARES_MAP);

        MPI_Gatherv(local_r__k.data(), items_per_worker, MPI_DOUBLE, r__k.data(), scounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(r__k.data(), (M + 1) * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        local_A___mul___r_k = local_multiply_matrix_on_vector(A, r__k);
        A___mul___r_k_norm = local_calc_euclidean_norm(local_A___mul___r_k);

        // TimeLog << "DEBUG: ";
        // for (int idx = 0; idx < local_r__k.size(); idx++)
        //     cout << local_r__k[idx] << " ";

        // cout << endl << endl;

        local_tau__k_inc_1 = calc_scalar_multiplication(local_A___mul___r_k, local_r__k) / (A___mul___r_k_norm * A___mul___r_k_norm);

        if (rank == 0) tau__k_inc_1 = 0.0;
        MPI_Reduce(&local_tau__k_inc_1, &tau__k_inc_1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&tau__k_inc_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        local_tau__k_inc_1___mul___r_k = multiply_vector_on_const(local_r__k, tau__k_inc_1);

        MPI_Scatterv(w__k.data(), scounts.data(), displs.data(), MPI_DOUBLE, local_w__k.data(), scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        local_w__k_inc_1 = subtract_vector_from_vector(local_w__k, local_tau__k_inc_1___mul___r_k);
        local_w__k_inc_1___sub___w_k = subtract_vector_from_vector(local_w__k_inc_1, local_w__k);
        
        // Условие остановки
        err = local_calc_euclidean_norm(local_w__k_inc_1___sub___w_k);

        if (rank == 0) {
            if ( iter % 2000 == 0 ) {
                TimeLog << "SLE resolving: err = " << err << endl;
            }

            if ( err < delta ) {
                break;
            }
        }

        MPI_Gatherv(local_w__k_inc_1.data(), scounts[rank], MPI_DOUBLE, w__k.data(), scounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(w__k.data(), (M + 1) * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

    }

    if (rank == 0) {
        double end_ts = MPI_Wtime();
        double total = end_ts - mpi_start_ts;
        TimeLog << "SUCCESSFULLY resolved SLE with <err = " << err << ", iterations_num = " << iter << ">" << "Time: " << total << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
