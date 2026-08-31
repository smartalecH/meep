/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

/* Relocatable array identity.
 *
 * BACKEND-PRIVATE header; not installed, never included from meep.hpp.
 *
 * Meep currently uses raw host addresses as both storage and identity: field
 * arrays, material arrays, DFT storage, polarization internals, the H == B
 * alias, metal-zero lists and boundary connections are all host pointers. A
 * plan built out of those addresses cannot be relocated to another backend, or
 * even survive a reallocation.
 *
 * These types give every array a stable name that is independent of where it
 * lives. A plan refers to (ArrayId, offset); exactly one object -- the CPU
 * array table -- knows what address an ArrayId currently resolves to.
 */

#ifndef MEEP_BACKEND_ARRAY_REF_HPP
#define MEEP_BACKEND_ARRAY_REF_HPP

#include <stddef.h>
#include <stdint.h>

namespace meep {

struct ArrayId {
  uint32_t value;
};

const uint32_t invalid_array_value = 0xffffffffu;
inline ArrayId invalid_array() { return ArrayId{invalid_array_value}; }
inline bool is_valid(ArrayId id) { return id.value != invalid_array_value; }
inline bool operator==(ArrayId a, ArrayId b) { return a.value == b.value; }
inline bool operator!=(ArrayId a, ArrayId b) { return a.value != b.value; }

/* What a value *means*. Required because Meep stores realnum,
   complex<realnum>, double, complex<double> and index arrays; a role plus an
   element count cannot size, copy, or validate storage on its own. Kept
   separate from Precision, which is how the value is *stored*: that separation
   is what lets a mixed-precision policy exist in Phase 2 without duplicating
   every descriptor. */
enum class ElementType { realnum_value, complex_realnum, float64, complex_float64, int32, index };

enum class Precision { f32, f64 }; // full PrecisionPolicy lands in PR 7

enum class array_role { field, material, polarization, dft, communication, scratch };

struct ArraySpec {
  ArrayId id;
  array_role role;
  ElementType element_type;
  Precision storage;  // CPU backend: native only. Must match across an alias pair.
  size_t elements;
  size_t alignment;
  ArrayId alias_of;   // invalid, or an explicit alias such as H == B
  /* Allocated in pass 1 and possibly elided after classify(); unused until
     PR 4, defined here so the descriptor does not change shape later. */
  bool classification_provisional;
  /* Classification resolves a provisional slot without renumbering it.
     Elided slots remain physically allocated until epoch retirement but are
     not part of the logical storage topology. */
  bool classification_elided;
};

struct ArrayRef {
  ArrayId id;
  size_t offset;
  size_t elements;
};

/* Regular index set. Preferred, and it covers the overwhelming majority of
   halos, metal-zero lists and monitor regions.

   Why slabs rather than one ElementRef per transferred real: an ElementRef is
   12-16 bytes against the 8 of the pointer it replaces, while preserving full
   per-element indirection. For a 3D chunk face at production resolution that is
   tens of megabytes of index metadata per halo and a purely gather-bound
   kernel. The underlying connections are almost always contiguous or
   constant-stride Yee slabs, so they are coalesced at construction and only the
   residue falls back to ElementRef. */
struct SlabRef {
  ArrayId array;
  ptrdiff_t base;
  int counts[3];        // 1 for absent dimensions
  ptrdiff_t strides[3];

  size_t elements() const {
    return size_t(counts[0]) * size_t(counts[1]) * size_t(counts[2]);
  }
};

struct ElementRef { // irregular fallback only
  ArrayId array;
  ptrdiff_t index;
};

enum class MemorySpace { host, pinned_host, device };

/* Opaque handle for "this buffer is ready". On CPU the transfer is complete
   when the comms_manager says so, and this carries no state; Phase 2 gives it
   an event. */
struct completion_token {
  uint32_t value;
};

struct CommBuffer {
  void *address;
  size_t elements;
  Precision element_precision; // buffers follow the sending array's storage precision
  MemorySpace space;
  completion_token ready;
};

} // namespace meep

#endif // MEEP_BACKEND_ARRAY_REF_HPP
