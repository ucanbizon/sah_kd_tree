#pragma once

/* SAH kd-tree construction algorithm implementation
 *
 * Copyright (c) 2019, Anatoliy V. Tomilov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following condition is met:
 * Redistributions of source code must retain the above copyright notice, this condition and the following disclaimer.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/functional.h>
#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/tuple.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/pointer.h>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>

#include <thrust/system/tbb/execution_policy.h>
#include <thrust/system/tbb/vector.h>

#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/cpp/vector.h>

#include <iterator>
#include <iostream>

#include <cstdio>
#include <cassert>

namespace kd_tree
{

using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

using I = int;
using U = unsigned int;
using F = float;

struct sah_params
{
    F reward_for_emptiness = 0.8f;
    F traversal_cost = 2.0f;
    F intersection_cost = 1.0f;
};

template< typename policy, template< typename ... > class container, I dimension >
struct projection
{
    using X = projection;
    using Y = projection< policy, container, (dimension + 1) % 3 >;
    using Z = projection< policy, container, (dimension + 2) % 3 >;

    const policy & p;

    // size is number of polygons
    struct
    {
        container< F > min, max; // AABB of polygon
        container< U > l, r; // indices to find corresponding event
        //container< U > triangle_index;
    } polygon;

    // size is number of nodes
    struct
    {
        container< F > min, max;
    } node;

    // size is number of events
    struct
    {
        container< F > pos;
        container< I > kind;
        container< U > polygon;

        container< U > l, r;

        container< U > node_index;

        container< I > split_kind;

        auto count() const { return pos.size(); }
    } event;

    // size is number of nodes
    struct
    {
        container< F > cost;
        container< F > pos;
        container< U > index;
        //container< I > split_count; // not needed
    } split;

    struct
    {
        container< U > triangle;
        container< U > node_index;
    } leaf;

    explicit
    projection(const policy & p, size_type triangle_count)
        : p{p}
    {
        polygon.min.resize(triangle_count);
        polygon.max.resize(triangle_count);
    }

    void generate_initial_events()
    {
        node.min.push_back(*thrust::min_element(p, polygon.min.cbegin(), polygon.min.cend()));
        node.max.push_back(*thrust::max_element(p, polygon.max.cbegin(), polygon.max.cend()));

        auto event_count = polygon.min.size() + polygon.max.size();

        event.pos.resize(event_count);
        event.kind.resize(event_count);
        event.polygon.resize(event_count);

        auto bbox_begin = thrust::make_zip_iterator(thrust::make_tuple(polygon.min.cbegin(), polygon.max.cbegin()));
        using bbox_type = typename decltype(bbox_begin)::value_type;
        auto event_begin = thrust::make_zip_iterator(thrust::make_tuple(event.pos.begin(), event.kind.begin(), event.polygon.begin()));
        auto event_end = thrust::make_zip_iterator(thrust::make_tuple(event.pos.end(), event.kind.end(), event.polygon.end()));
        auto twice = [] __host__ __device__ (U index) -> U { return index + index; };
        auto event_pair_index_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(0u), twice);
        auto event_pair_begin = thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(event_begin, thrust::next(event_begin))), event_pair_index_begin);
        using event_pair_type = typename decltype(event_pair_begin)::value_type;
        auto polygon_bbox_begin = thrust::make_zip_iterator(thrust::make_tuple(bbox_begin, thrust::make_counting_iterator(0u)));
        using polygon_bbox_type = typename decltype(polygon_bbox_begin)::value_type;
        auto to_event = [] __host__ __device__ (polygon_bbox_type polygon_bbox) -> event_pair_type
        {
            const auto & bbox = thrust::get< 0 >(polygon_bbox);
            F min = thrust::get< 0 >(bbox), max = thrust::get< 1 >(bbox);
            U event_polygon = thrust::get< 1 >(polygon_bbox);
            return {{min, (min < max) ? +1 : 0, event_polygon}, {max, -1, event_polygon}};
        };
        auto begin = thrust::make_transform_iterator(polygon_bbox_begin, to_event);
        auto event_rbegin = thrust::make_reverse_iterator(event_end);
        auto planar_event_begin = thrust::make_zip_iterator(thrust::make_tuple(event_rbegin, thrust::make_discard_iterator()));
        auto is_planar_event = [] __host__ __device__ (bbox_type bbox) -> bool { return thrust::get< 0 >(bbox) < thrust::get< 1 >(bbox); };
        auto ends = thrust::stable_partition_copy(p, begin, thrust::next(begin, event_count / 2), bbox_begin, planar_event_begin, event_pair_begin, is_planar_event);
        auto non_planar_event_count = size_type(2 * std::distance(event_pair_begin, ends.second));
        if (non_planar_event_count < event_count) {
            auto planar_event_count = (event_count - non_planar_event_count) / 2;
            event_count -= planar_event_count;

            thrust::copy(p, event_rbegin, thrust::next(event_rbegin, planar_event_count), thrust::next(event_begin, non_planar_event_count));

            event.pos.resize(event_count);
            event.kind.resize(event_count);
            event.polygon.resize(event_count);
        } else {
            assert(non_planar_event_count == event_count);
        }

        thrust::sort(p, event_begin, thrust::next(event_begin, event_count));
    }

    void calculate_triangle_counts()
    {
        auto event_count = event.count();

        event.node_index.resize(event_count, 0u);

        event.l.resize(event_count);
        event.r.resize(event_count);

        auto l_begin = thrust::make_transform_iterator(event.kind.cbegin(), [] __host__ __device__ (I type) -> U { return (type < 0) ? 0u : 1u; });
        thrust::exclusive_scan_by_key(p, event.node_index.cbegin(), event.node_index.cend(), l_begin, event.l.begin());
        auto r_begin = thrust::make_transform_iterator(event.kind.crbegin(), [] __host__ __device__ (I type) -> U { return (type > 0) ? 0u : 1u; });
        thrust::exclusive_scan_by_key(p, event.node_index.crbegin(), event.node_index.crend(), r_begin, event.r.rbegin());
    }

    void find_perfect_splits(const sah_params & sah, const Y & y, const Z & z)
    {
        auto node_count = size_type(event.node_index.back());
        ++node_count;

        split.cost.resize(node_count);
        split.pos.resize(node_count);
        split.index.resize(node_count);

        auto perfect_split_begin = thrust::make_zip_iterator(thrust::make_tuple(split.cost.begin(), split.pos.begin(), split.index.begin()));
        using perfect_split_type = typename decltype(perfect_split_begin)::value_type;
        auto node_limits_begin = thrust::make_zip_iterator(thrust::make_tuple(node.min.cbegin(), node.max.cbegin(), y.node.min.cbegin(), y.node.max.cbegin(), z.node.min.cbegin(), z.node.max.cbegin()));
        auto node_box_begin = thrust::make_permutation_iterator(node_limits_begin, event.node_index.cbegin());
        auto split_index_begin = thrust::make_counting_iterator< U >(0u);
        auto sah_args_begin = thrust::make_zip_iterator(thrust::make_tuple(node_box_begin, event.pos.cbegin(), event.kind.cbegin(), split_index_begin, event.l.cbegin(), event.r.cbegin()));
        using sah_args_type = typename decltype(sah_args_begin)::value_type;
        auto to_sah = [sah] __host__ __device__ (sah_args_type sah_args) -> perfect_split_type
        {
            const auto & node_box = thrust::get< 0 >(sah_args);
            F event_pos = thrust::get< 1 >(sah_args);
            I event_kind = thrust::get< 2 >(sah_args);
            U split_index = thrust::get< 3 >(sah_args);
            U l = thrust::get< 4 >(sah_args), r = thrust::get< 5 >(sah_args);
            F min = thrust::get< 0 >(node_box), max = thrust::get< 1 >(node_box);
            F L = event_pos - min;
            F R = max - event_pos;
            if (event_kind < 0) {
                ++split_index;
            } else if (event_kind == 0) {
                if (L < R) {
                    ++l;
                    ++split_index;
                } else {
                    ++r;
                }
            }
            F x = max - min;
            F y = thrust::get< 3 >(node_box) - thrust::get< 2 >(node_box);
            F z = thrust::get< 5 >(node_box) - thrust::get< 4 >(node_box);
            F xx = y + z;
            F yz = y * z;
            F split_cost = (l * (yz + xx * L) + r * (yz + xx * R)) / (yz + xx * x);
            split_cost *= sah.intersection_cost;
            split_cost += sah.traversal_cost;
            if ((l == 0) || (r == 0)) {
                split_cost *= sah.reward_for_emptiness;
            }
            return {split_cost, event_pos, split_index};
        };
        auto sah_begin = thrust::make_transform_iterator(sah_args_begin, to_sah);
        thrust::reduce_by_key(p, event.node_index.cbegin(), event.node_index.cend(), sah_begin, thrust::make_discard_iterator(), perfect_split_begin, thrust::equal_to< U >{}, thrust::minimum< perfect_split_type >{});
    }

    void set_event_counterparts()
    {
        auto event_count = event.count();

        polygon.l.resize(polygon.min.size());
        polygon.r.resize(polygon.max.size());

        auto event_index_begin = thrust::make_counting_iterator(0u);
        auto event_index_end = thrust::next(event_index_begin, event_count);
        thrust::scatter_if(p, event_index_begin, event_index_end, event.polygon.cbegin(), event.kind.cbegin(), polygon.l.begin(), [] __host__ __device__ (I kind) -> bool { return !(kind < 0); });
        thrust::scatter_if(p, event_index_begin, event_index_end, event.polygon.cbegin(), event.kind.cbegin(), polygon.r.begin(), [] __host__ __device__ (I kind) -> bool { return !(kind > 0); });
    }

    template< typename split_indices_type, typename event_indices_type >
    static
    __forceinline__ __host__ __device__
    I get_event_split_kind(const split_indices_type & split_indices, const event_indices_type & event_indices)
    {
        U split_index = thrust::get< dimension >(split_indices);
        U l = thrust::get< 0 >(thrust::get< dimension >(event_indices));
        U r = thrust::get< 1 >(thrust::get< dimension >(event_indices));
        if (r < split_index) {
            return -1; // left
        } else if (l < split_index) {
            return 0; // splitted
        } else {
            return +1; // right
        }
    }

    void split_triangles(const container< I > & best_dimension, const Y & y, const Z & z)
    {
        auto event_count = event.count();

        { // separate completed nodes
            auto best_dimension_begin = thrust::make_permutation_iterator(best_dimension.cbegin(), event.node_index.cbegin());
            auto leaf_triangle_begin = thrust::make_zip_iterator(thrust::make_tuple(best_dimension_begin, event.kind.cbegin()));
            using leaf_triangle_type = typename decltype(leaf_triangle_begin)::value_type;
            auto is_leaf_triangle = [] __host__ __device__ (leaf_triangle_type leaf_triangle) -> bool
            {
                I best_dimension = thrust::get< 0 >(leaf_triangle);
                U event_kind = thrust::get< 1 >(leaf_triangle);
                return (best_dimension < 0) && !(0 < event_kind);
            };
            auto leaf_triangle_count = thrust::count_if(p, leaf_triangle_begin, thrust::next(leaf_triangle_begin, event_count), is_leaf_triangle);
            if (leaf_triangle_count > 0) {
                auto leaf_size = leaf.triangle.size();
                assert(leaf_size == leaf.node_index.size());
                leaf.triangle.resize(leaf_size + leaf_triangle_count);
                leaf.node_index.resize(leaf_size + leaf_triangle_count);
                auto event_begin = thrust::make_zip_iterator(thrust::make_tuple(event.polygon.cbegin(), event.node_index.cbegin()));
                //using event_type = typename decltype(event_begin)::value_type;
                auto leaf_begin = thrust::make_zip_iterator(thrust::make_tuple(leaf.triangle.begin(), leaf.node_index.begin()));
                thrust::copy_if(p, event_begin, thrust::next(event_begin, event_count), leaf_triangle_begin, thrust::next(leaf_begin, leaf_size), is_leaf_triangle);
            }
        }

        { // get split kind for each event
            event.split_kind.resize(event_count, 0);

            auto split_indices_begin = thrust::make_zip_iterator(thrust::make_tuple(split.index.cbegin(), y.split.index.cbegin(), z.split.index.cbegin()));
            auto event_begin = thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(best_dimension.cbegin(), split_indices_begin)), event.node_index.cbegin());
            using event_type = typename decltype(event_begin)::value_type;
            auto x_event_indices_begin = thrust::make_zip_iterator(thrust::make_tuple(polygon.l.cbegin(), polygon.r.cbegin()));
            auto y_event_indices_begin = thrust::make_zip_iterator(thrust::make_tuple(y.polygon.l.cbegin(), y.polygon.r.cbegin()));
            auto z_event_indices_begin = thrust::make_zip_iterator(thrust::make_tuple(z.polygon.l.cbegin(), z.polygon.r.cbegin()));
            auto event_indices_begin = thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(x_event_indices_begin, y_event_indices_begin, z_event_indices_begin)), event.polygon.cbegin());
            using event_indices_type = typename decltype(event_indices_begin)::value_type;
            auto to_split_kind = [] __host__ __device__ (event_type event, event_indices_type event_indices) -> I
            {
                I best_dimension = thrust::get< 0 >(event);
                const auto & split_indices = thrust::get< 1 >(event);
                switch (best_dimension) {
                case dimension : {
                    return X::get_event_split_kind(split_indices, event_indices);
                }
                case (dimension + 1) % 3 : {
                    return Y::get_event_split_kind(split_indices, event_indices);
                }
                case (dimension + 2) % 3 : {
                    return Z::get_event_split_kind(split_indices, event_indices);
                }
                default : {
                    return 0;
                }
                }
            };
            thrust::transform(p, event_begin, thrust::next(event_begin, event_count), event_indices_begin, event.split_kind.begin(), to_split_kind);
        }

        // TODO: mark events by node index
        // TODO: partition events into left/right and splitted/completed separate arrays

        // TODO: split triangles
    }
};

template< template< typename ... > class container, typename policy, typename input_iterator >
void build(const policy p, const sah_params & sah, input_iterator b, input_iterator e)
{
    auto triangle_count = size_type(thrust::distance(b, e));

    projection< policy, container, 0 > x{p, triangle_count};
    projection< policy, container, 1 > y{p, triangle_count};
    projection< policy, container, 2 > z{p, triangle_count};

    {
        auto triangle_box_begin = thrust::make_zip_iterator(thrust::make_tuple(x.polygon.min.begin(), y.polygon.min.begin(), z.polygon.min.begin(), x.polygon.max.begin(), y.polygon.max.begin(), z.polygon.max.begin()));
        using triangle_box_type = typename decltype(triangle_box_begin)::value_type;
        using T = typename thrust::iterator_traits< input_iterator >::value_type;
        auto to_triangle_box = [] __host__ __device__ (T t) -> triangle_box_type
        {
            auto min = fminf(fminf(t.A, t.B), t.C), max = fmaxf(fmaxf(t.A, t.B), t.C);
            return {min.x, min.y, min.z, max.x, max.y, max.z};
        };
        thrust::transform(p, b, e, triangle_box_begin, to_triangle_box);
    }

    x.generate_initial_events();
    y.generate_initial_events();
    z.generate_initial_events();

    container< I > best_dimension;
    container< I > polygon_count;
    polygon_count.push_back(I(triangle_count));

    for (;;) {
        x.calculate_triangle_counts();
        y.calculate_triangle_counts();
        z.calculate_triangle_counts();

        x.find_perfect_splits(sah, y, z);
        y.find_perfect_splits(sah, z, x);
        z.find_perfect_splits(sah, x, y);

        auto split_cost_begin = thrust::make_zip_iterator(thrust::make_tuple(x.split.cost.cbegin(), y.split.cost.cbegin(), z.split.cost.cbegin()));
        using split_cost_type = typename decltype(split_cost_begin)::value_type;
        auto to_best_dimension = [sah] __host__ __device__ (split_cost_type split_cost, I polygon_count) -> I
        {
            F x = thrust::get< 0 >(split_cost);
            F y = thrust::get< 1 >(split_cost);
            F z = thrust::get< 2 >(split_cost);
            F best_split_cost = fminf(sah.intersection_cost * polygon_count, fminf(x, fminf(y, z)));
            if (!(x > best_split_cost)) {
                return 0;
            } else if (!(y > best_split_cost)) {
                return 1;
            } else if (!(z > best_split_cost)) {
                return 2;
            } else {
                return -1;
            }
        };
        auto node_count = polygon_count.size();
        best_dimension.resize(node_count);
        thrust::transform(split_cost_begin, thrust::next(split_cost_begin, node_count), polygon_count.cbegin(), best_dimension.begin(), to_best_dimension);

        x.set_event_counterparts();
        y.set_event_counterparts();
        z.set_event_counterparts();

        x.split_triangles(best_dimension, y, z);
        y.split_triangles(best_dimension, z, x);
        z.split_triangles(best_dimension, x, y);

        // TODO: calculate bboxes for nodes

        break; // TODO: close the loop
    }
}

}
