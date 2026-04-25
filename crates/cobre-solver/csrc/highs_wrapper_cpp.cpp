/* C++ shim for cobre_highs_set_basis_non_alien.
 *
 * This file is compiled as C++17 and exposes a single function with C linkage.
 * It exists as a separate translation unit because the main wrapper
 * (highs_wrapper.c) is compiled as plain C and cannot use C++ types such as
 * HighsBasis directly.
 *
 * The function constructs a HighsBasis with alien = false before calling
 * Highs::setBasis(const HighsBasis&).  This bypasses the throwaway LU
 * factorisation that the HiGHS C API (Highs_setBasis) triggers when alien is
 * true.  Callers must guarantee that the total number of kBasic entries equals
 * num_row; the function returns kHighsStatusError if isBasisConsistent fails
 * inside HiGHS.
 */

#include "highs_wrapper.h"
#include <Highs.h>

extern "C" {

int32_t cobre_highs_set_basis_non_alien(
    void*           highs,
    const int32_t*  col_status,
    const int32_t*  row_status
) {
    Highs* h = reinterpret_cast<Highs*>(highs);

    const HighsInt num_col = h->getNumCol();
    const HighsInt num_row = h->getNumRow();

    HighsBasis basis;
    basis.alien = false;
    basis.valid = true;
    basis.col_status.resize(static_cast<std::size_t>(num_col));
    basis.row_status.resize(static_cast<std::size_t>(num_row));

    for (HighsInt i = 0; i < num_col; ++i) {
        basis.col_status[static_cast<std::size_t>(i)] =
            static_cast<HighsBasisStatus>(col_status[i]);
    }

    for (HighsInt i = 0; i < num_row; ++i) {
        basis.row_status[static_cast<std::size_t>(i)] =
            static_cast<HighsBasisStatus>(row_status[i]);
    }

    return static_cast<int32_t>(h->setBasis(basis));
}

} /* extern "C" */
