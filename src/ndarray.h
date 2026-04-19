// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
// Simple N-dimensional array type with control of allocation.
#ifndef AWAVE_NDARRAY_H
#define AWAVE_NDARRAY_H

#include <array>
#include <concepts>
#include <memory>
#include <numeric>

namespace nd {

    // Fixed size array with elementwise operations.
    template <typename T, std::size_t N>
    struct vec : std::array<T, N> {

        friend bool operator==(vec const& l, vec const& r) {
            bool ans = true;
            for (auto i = 0u; i < N; ++i)
                ans &= (l[i] == r[i]);
            return ans;
        }

        friend vec operator+(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = l[i] + r[i];
            return ans;
        }
        friend vec operator-(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = l[i] - r[i];
            return ans;
        }

        friend vec operator+(vec const& v, T scalar) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = v[i] + scalar;
            return ans;
        }
        friend vec operator-(vec const& v, T scalar) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = v[i] - scalar;
            return ans;
        }
        friend vec operator*(vec const& v, T scalar) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = v[i] * scalar;
            return ans;
        }
        friend vec operator/(vec const& v, T scalar) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = v[i] / scalar;
            return ans;
        }

        friend vec operator*(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = l[i] * r[i];
            return ans;
        }
        friend vec operator/(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = l[i] / r[i];
            return ans;
        }
        friend vec min(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
              ans[i] = std::min(l[i], r[i]);
            return ans;
        }
        friend vec max(vec const& l, vec const& r) {
            vec ans;
            for (auto i = 0u; i < N; ++i)
                ans[i] = std::max(l[i], r[i]);
            return ans;
        }
    };

    template <unsigned N>
    using index = vec<std::size_t, N>;

    template <typename T>
    using host_allocator = std::allocator<T>;

    // Base for arrays and views that does the indexing but not
    // resource management
    template <typename T, int N>
    class base {
    public:
        static constexpr int ND = N;
        using value_type = T;
        using reference = T&;
        using const_reference = T const&;
        using pointer = T*;
        using const_pointer = T const*;

        using index_type = index<ND>;

    protected:
        static constexpr std::size_t make_strides_size(index_type& strides, index_type const& shape) {
            strides[N - 1] = 1;
            for (int i = N - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides[0] * shape[0];
        }
        static constexpr std::size_t make_index(index_type const& idx, index_type const& strides) {
            std::size_t result = idx[N - 1];
            // We can skip N-1 as that's always == 1
            for (int i = 0; i < N - 1; ++i)
                result += idx[i] * strides[i];
            return result;
        }

        // Ensure value initialised to zero
        index_type m_shape = {};
        index_type m_strides = {};
        std::size_t m_size = 0;
        T* m_data = nullptr;

        base() = default;
        base(index_type shape, T* data = nullptr) : m_shape{shape}, m_data{data} {
            m_size = make_strides_size(m_strides, m_shape);
        }
        base(index_type shape, index_type strides, std::size_t size, T* data) :
          m_shape{shape}, m_strides{strides}, m_size{size}, m_data{data}
        {
        }

    public:
        constexpr auto size() const {
            return m_size;
        }

        constexpr auto const& shape() const {
            return m_shape;
        }

        constexpr auto const& strides() const {
            return m_strides;
        }

        // Mutable raw data access
        constexpr pointer data() {
            return m_data;
        }

        // Const raw data access
        constexpr const_pointer data() const {
            return m_data;
        }

        // Mutable element access
        // By ND index
        constexpr reference operator[](index_type ind) {
            auto i = make_index(ind, m_strides);
            return m_data[i];
        }

        // By multiple indices
        template <typename... Ints>
        requires (sizeof...(Ints) == ND)
        constexpr reference operator()(Ints... inds) {
            return operator[](index_type{inds...});
        }

        constexpr const_reference operator[](index_type ind) const {
            auto i = make_index(ind, m_strides);
            return m_data[i];
        }

        // Const element access
        template <typename... Ints>
        requires (sizeof...(Ints) == ND)
        constexpr const_reference operator()(Ints... inds) const {
            return operator[](index_type{inds...});
        }

    };

    template <typename T, int N, typename Allocator>
    class array;

    // Non-owning view over data, a bit like an mdspan
    // Construct one from an array
    // Can be freely copied but will be invalidated by changes to the
    // underlying array
    template <typename T, int N>
    class view : public base <T, N> {
    public:
        using base_type = base<T, N>;

    private:
        template <typename, int, typename>
        friend class array;
        using base_type::base_type;
    };

    // Simple multidimensional array class
    // - Contained type must be trivial
    // - Initial contents uninitialised
    // - Uniquely owns its data
    template <typename T, int N, typename Allocator = host_allocator<T>>
    class array : public base <T, N> {
    public:
        using base_type = base <T, N>;
        using index_type = base_type::index_type;
        using allocator_type = Allocator;
        using view_type = view<T, N>;

        array() = default;

        explicit array(const index_type& xs) : base_type{xs} {
            this->m_data = allocator_type().allocate(this->m_size);
        }

        template <std::integral... Ints>
        requires(sizeof...(Ints) == array::ND)
        explicit array(Ints... shape) : array(index_type{typename index_type::value_type(shape)...}) {
        }

        // No copying
        array(const array&) = delete;

        array& operator=(const array&) = delete;

        // Moving fine, do the swap pattern
        array(array&& src) noexcept {
            swap(*this, src);
        }

        array& operator=(array&& src) noexcept {
            swap(*this, src);
            return *this;
        }

        // Destructor
        ~array() {
            allocator_type().deallocate(this->m_data, this->m_size);
        }

        constexpr friend void swap(array& lhs, array& rhs) noexcept {
            using namespace std;
            swap(lhs.m_shape, rhs.m_shape);
            swap(lhs.m_strides, rhs.m_strides);
            swap(lhs.m_size, rhs.m_size);
            swap(lhs.m_data, rhs.m_data);
        }

        view_type get_view() {
          auto& self = *this;
          return view_type{self.m_shape, self.m_strides, self.m_size, self.m_data};
        }

        operator view_type() {
            return get_view();
        }

        // Return a copy
        template <typename F = std::nullptr_t>
        array clone(F&& copy_func = nullptr) const {
            auto ans = array{this->m_shape};
            if constexpr(std::same_as<F, std::nullptr_t>) {
                std::copy(this->m_data, this->m_data + this->m_size, ans.data());
            } else {
                copy_func(this->m_data, this->m_data + this->m_size, ans.data());
            }
            return ans;
        }
    };

}

// Make nd::vec work with structured bindings by implementing the tuple interface
namespace std {
    template <typename T, size_t N>
    struct tuple_size<nd::vec<T, N>> : integral_constant<size_t, N> {};
    template<size_t I, typename T, size_t N>
    struct tuple_element<I, nd::vec<T, N>> {
        using type = T;
    };
}
#endif
