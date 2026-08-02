#include "microkernel/dtype.h"
#include <stdexcept>
#include <cstring>

size_t element_size(DType dtype) {

    switch(dtype) {

        case DType::Float32 : return 4;
        case DType::Float16 : return 2;
        case DType::Int8 : return 1;
        case DType::Int4 : return 1;
        default : throw std::invalid_argument("Unknown DType."); 
    }

}


std::string dtype_to_string(DType dtype) {

    switch(dtype) {

        case DType::Float32 : return "Float32";
        case DType::Float16 : return "Float16";
        case DType::Int8 : return "Int8";
        case DType::Int4 : return "Int4";
        default : throw std::invalid_argument("Unknown DType."); 
    }

}

uint16_t float_to_half_bits(const uint32_t value) noexcept {

    // 1. Extract sign (bit 31), exponent (bits 30-23, 8 bits), mantissa (bits 22-0, 23 bits)
    //    via mask + shift.

    // shift sign (bit 31) to bit 0, and all bits above get filled with 0 (right shift), no mask needed.

    uint32_t sign32 = value >> 31;

    // shift exponent (bits 30-23) to bit 0, and mask all bits on top of the 8th bit (31-23=8).
    // & 0xFF keeps only the lowest 8 bits (1 F corrisponds to 1111).
    uint32_t exp32 = value >> 23 & 0xFF;

    // no shift needed here, we keep the lowest 23 bits
    uint32_t mant32 = value & 0x7FFFFF;


    // 2. Check for exp32 == 0xFF (all ones) -> source is Inf or NaN.
    //    - mant32 == 0  -> Inf: sign | (0x1F << 10)
    //    - mant32 != 0  -> NaN: sign | (0x1F << 10) | nonzero mantissa bits
    //      (make sure the shifted-down mantissa can't become 0, or NaN silently becomes Inf)
    //    Return early in both cases.

    if (exp32 == 0xFF) {

        if (mant32==0){

            // Inf case. Build the 16-bit pattern from two fields, each shifted to its own
            // position, combined with |, then return it (implicit narrowing uint32_t -> uint16_t):
            //   - sign32 shifted to bit 15
            //   - the 5-bit all-ones exponent (0x1F) shifted to bits 14-10
            // No mantissa term -- it's 0 for Inf.

            uint32_t sign16 = sign32 << 15;
            uint32_t exp16 = 0x1F << 10;
            return sign16 | exp16;
        }

        else {

            // NaN case. Same two terms as Inf, plus a third for the mantissa:
            //   - shift mant32 down from 23 bits to 10 bits (same width reduction as step 6)
            //   - that shift alone can produce 0 for some inputs (e.g. if only high mantissa
            //     bits were set and they land outside the top 10) 
            uint32_t sign16 = sign32 << 15;
            uint32_t exp16 = 0x1F << 10;
            // 0x200 = bit 9, the top bit of the 10-bit mantissa field. OR-ing it in guarantees
            // the mantissa term is never all-zero (which would wrongly collapse to Inf), and by
            // convention setting the top mantissa bit marks this as a quiet NaN.
            uint32_t mant16 = (mant32 >> 13) | 0x200;

            return sign16 | exp16 | mant16;
        }

    }


    // 3. Rebias the exponent: new_exp = exp32 - 127 + 15. Signed type, this can go negative.

    int32_t new_exp = static_cast<int32_t>(exp32) - 127 + 15;

    // 4. Overflow: new_exp >= 0x1F -> result doesn't fit in half range, return signed Inf.

    if(new_exp >= 0x1F) {

        // return inf
        uint32_t sign16 = sign32 << 15;
        uint32_t exp16 = 0x1F << 10;
        return sign16 | exp16;

    }

    // 5. Underflow / subnormal: new_exp <= 0 -> result is a half subnormal or zero.

    if (new_exp <=0) {

        // a) "Significand" (aka "mantissa" in the loose/common sense) is the digit-string part of
        //    a number in scientific notation -- the "1.10" in "1.10 x 2^2", as opposed to the exponent.
        //    mant32 only holds the fractional part *after* the implicit "1." (that's what float32
        //    actually stores). Here we rebuild the true, explicit significand as a plain 24-bit
        //    integer: OR a 1-bit at position 23 (the implicit leading 1) together with the 23
        //    fraction bits already in mant32. The result, significand24, is the real numeric
        //    significand this float represents, with nothing left implicit -- necessary because
        //    half's subnormal encoding has no implicit bit of its own to lean on; every bit must be
        //    physically present before we shift it into position.

        uint32_t leading_one = 1 << 23;
        uint32_t significand24 = mant32 | leading_one;


        // b) Compute the shift amount into a NEW variable (e.g. `shift`) -- don't overwrite new_exp,
        //    step 6 still needs its original value.
        //
        //    Why 14 - new_exp: significand24 currently represents the value
        //    significand24 x 2^(new_exp - 24) (24 fractional bits, scaled by the original exponent
        //    minus the width of the field). Half's subnormal encoding represents a value as
        //    mantissa_h x 2^-24 (10-bit field, fixed exponent -14, i.e. total scale 2^(-10-14)).
        //    Setting those equal and solving for how far to shift significand24 right to land on
        //    mantissa_h gives shift = 14 - new_exp. 

        uint32_t shift = 14 - new_exp;

        // c) Guard: if shift >= 24, every bit of the 24-bit significand shifts out -> the value
        //    underflows completely. Return just the sign bit (signed zero), no exponent/mantissa term

        if (shift >= 24) {

            return sign32 << 15;

        }

        // d) Otherwise: mantissa_h = significand >> shift. Exponent field is 0 (that's what marks
        //    this as subnormal), so the result is just sign16 | mantissa_h -- no exponent term to OR in.

        uint32_t sign16 = sign32 << 15;
        uint32_t mantissa_h = significand24 >> shift;
        return sign16 | mantissa_h;

    }

    // 6. Normal case (0 < new_exp < 0x1F): pack sign | (new_exp << 10) | (mant32 >> 13).
    //    Right-shifting 23 -> 10 bits truncates; decide/document if you're rounding or truncating.

    return (sign32 << 15) | (new_exp << 10) | (mant32 >> 13);

}

uint32_t half_to_float_bits(const uint16_t value) noexcept {

    // Mirror image of float_to_half_bits: extract half's three fields, then rebuild them
    // at float32's wider widths.

    // 1. Extract sign16 (bit 15), exp16 (bits 14-10, 5 bits), mant16 (bits 9-0, 10 bits).
    //    Same mask+shift idiom as before, just different bit positions/widths.

    
    uint32_t sign16 = value >> 15;
    uint32_t exp16 = value >> 10 & 0x1F;
    uint32_t mant16 = value & 0x3FF;


    // 2. Check exp16 == 0x1F (all ones) -> source is half Inf or NaN.
    //    - mant16 == 0 -> Inf: sign shifted to bit 31 | float32's all-ones exponent (0xFF) at bits 30-23
    //    - mant16 != 0 -> NaN: same, plus mant16 widened from 10 bits to 23 bits: shift left by
    //      23 - 10 = 13 (exact mirror of the >> 13 truncation used earlier).

    if (exp16 == 0x1F) {

        // inf case.
        if (mant16 == 0) {

            return (sign16 << 31) | (0xFF << 23);

        } 
        
        // NaN case.
        else {

            return (sign16 << 31) | (0xFF << 23) | (mant16 << 13); 

        }


    }

    // 3. Check exp16 == 0 (subnormal or zero source).
    //    - mant16 == 0 -> signed zero (sign shifted to bit 31, nothing else).
    //    - mant16 != 0 -> subnormal half. This needs normalization: mant16 doesn't have an implicit
    //      leading 1 (that's what subnormal means), so its true magnitude depends on where its
    //      highest set bit is. Shift mant16 left until that highest bit reaches the implicit-1
    //      position, counting how many shifts that took -- each shift left corresponds to decreasing
    //      the effective exponent by 1 from the smallest half exponent. Work out the starting
    //      exponent value and loop condition on paper first; this is the trickiest part, mirroring
    //      the subnormal case from the other direction.

    if (exp16 == 0) {

        if (mant16 == 0) {

            return sign16 << 31;

        }
        
        else {

             // Normalize: shift mant16 left, one bit at a time, until its highest set bit lands
             // at bit 23 (float32's implicit-1 position, same convention as significand24 in the
             // encode function). `shift` counts how many left-shifts that took.
             int shift {0};

             while (!(mant16 & 0x800000)) {

                shift++;

                mant16 = mant16 << 1;

             };

             // Each shift needed to reach bit 23 corresponds to the result's exponent being one
             // less than a fixed baseline (derived algebraically, verified against the smallest
             // half subnormal 2^-24 and the largest, 2^-15 -- see the worked examples).
             uint32_t exp32 = 126 - shift;

             // mant16 now has bit 23 set (the implicit leading 1) plus whatever's below it --
             // mask that bit off, keeping only bits 22-0: exactly float32's 23-bit fraction.
             uint32_t fraction = mant16 & 0x7FFFFF;

             return (sign16 << 31) | (exp32 << 23) | fraction;
        
        }

    }

    // 4. Normal case: new_exp = exp16 - 15 + 127, mant32 = mant16 shifted left by 13 (widening,
    //    opposite direction of the truncation in float_to_half_bits step 6).
    //    Pack and return: sign | (new_exp << 23) | mant32.

    uint32_t new_exp = exp16 - 15 + 127;
    uint32_t mant32 = mant16 << 13;
    return sign16 | (new_exp << 23) | mant32;

}

Half::Half(float value) {

    // value's bytes, reinterpreted as a uint32_t, not converted: memcpy copies the raw 4 bytes
    // of the float `value` into `value_int` unchanged. This is NOT the same as a numeric cast
    // like (uint32_t)value, which would round the float to an integer (3.14f -> 3) -- we want
    // the identical bit pattern, just accessible as an integer we can shift/mask, which is what
    // float_to_half_bits expects as input.
    uint32_t value_int;

    std::memcpy(&value_int, &value, sizeof(value_int));

    bits = float_to_half_bits(value_int);

}

Half::operator float() const {

    uint32_t value_int;

    value_int = half_to_float_bits(bits);

    float return_value;

    std::memcpy(&return_value, &value_int, sizeof(float));

    return return_value;

}

int8_t unpack_int4(std::byte b, bool high_nibble) {

    uint8_t working_int = std::to_integer<uint8_t>(b);

    if(high_nibble){

        // High nibble lives in bits 7-4. Mask with 0xF0 to zero out the low nibble (bits 3-0,
        // which belong to the other packed element, not this one), then shift right by 4 to
        // bring those bits down to positions 3-0.
        working_int = (working_int & 0xF0) >> 4;

    }

    else {

        // Low nibble lives in bits 3-0, already at the bottom.
        working_int = (working_int & 0x0F);

    }

    // At this point working_int is a raw 0-15 value -- correct magnitude, but not yet a valid
    // signed number. Int4 is signed two's complement (-8..7): values 0-7 are already correct
    // as-is, but 8-15 are meant to represent -8..-1 and currently read as positive (e.g. 15
    // should be -1, but right now it reads as plain 15).
    //
    // Note this is NOT sign-magnitude (a separate sign bit + a magnitude, e.g. -1 = "1" + "001").
    // C++ signed integers (int8_t included) are two's complement: the top bit has a NEGATIVE
    // place value instead of a dedicated sign flag. In 4 bits that means place values
    // (-8, 4, 2, 1), so 1111 = -8+4+2+1 = -1, and 1000 = -8+0+0+0 = -8. This is also why
    // 1 + (-1) works out to 0 using plain binary addition with the overflow bit discarded --
    // two's complement is what lets hardware add/subtract signed numbers with the exact same
    // circuitry as unsigned, no separate sign handling required.
    //
    // Sign extension fixes the width mismatch: check whether bit 3 (the 4-bit value's own sign
    // bit) is set, and if so, propagate that sign into the upper bits of the 8-bit result --
    // either by explicitly OR-ing in 0xF0 when bit 3 is set, or via the
    // shift-left-4-then-arithmetic-shift-right-4 trick discussed earlier. Return as int8_t.


    if ( working_int & 0x8 ) {

        working_int = working_int | 0xF0;

    }

    return working_int;

}

std::byte pack_int4(std::byte b, bool high_nibble, int8_t value) {

    // reject values that don't fit in signed 4 bits
    if (value < -8 || value > 7) {

        throw std::out_of_range("Value " + std::to_string(value) + " is outside the representable int4" +
        " range [-8, 7].");

    }

    uint8_t working_value = static_cast<uint8_t>(value) & 0x0F;

    // existing byte, as a plain integer we can mask/shift
    uint8_t working_int = std::to_integer<uint8_t>(b);

    if(high_nibble) {

        // clear bits 7-4, OR in the new nibble shifted into place
        working_int = (working_int & 0x0F) | (working_value << 4);

    }

    else {

        // clear bits 3-0, OR in the new nibble (already in place, no shift)
        working_int = (working_int & 0xF0) | (working_value);

    }

    // back to std::byte for the return type
    return static_cast<std::byte>(working_int);

}