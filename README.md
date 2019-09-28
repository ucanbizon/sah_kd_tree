```cpp

    #include <vector_functions.hpp>
    #include <helper_math.h>
    
    using Vertex = float3;    
    struct Triangle { Vertex A, B, C; };
	
        kd_tree::sah_params sah{/*...*/};
        thrust::cuda::pointer< const Triangle > b{triangles.cbegin()}, e{triangles.cend()};
		switch (thrust_backend) {
        case 0 : {
            kd_tree::build< thrust::cuda::vector >(thrust::cuda::par.on(stream.getStream()), sah, b, e);
			break;
        }
		case 1 : {
            kd_tree::build< thrust::tbb::vector >(thrust::tbb::par, sah, b, e);
			break;
        }
		case 2 : {
            kd_tree::build< thrust::omp::vector >(thrust::omp::par, sah, b, e);
			break;
        }
		case 3 : {
            kd_tree::build< thrust::cpp::vector >(thrust::cpp::par, sah, b, e);
			break;
        }
		case 4 : {
            kd_tree::build< thrust::host_vector >(thrust::seq, sah, b, e);
			break;
        }
		}
