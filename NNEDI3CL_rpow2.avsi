### A wrapper of NNEDI3CL for enlarging images by powers of 2.


### Requirements - AviSynth+ r3688 or later, NNEDI3CL, avsresize.


### Usage ###
# NNEDI3CL_rpow2(clip input, int "rfactor", int "field", int "nsize", int "nns", int "qual", int "etype", int "pscrn", int "device", bool "st", string "cshift", string "chroma_loc", bool "luma")
###
## Parameters ##
#---------------
# rfactor (default 2): Image enlargement factor.
# Must be a power of 2.
#---------------
# cshift (default not specified): Sets the resizer used for correcting the image center shift.
# If not specified the shift is not corrected.
# For example, cshift="spline36".
#---------------
# chroma_loc (default "left"): Chroma location if cshift is specified.
# Must be either "left" or "center".


### Version: 1.0.3


### Changelog ###
#---------------
# Added parameter luma. (NNEDI3CL 1.0.7)
#---------------
# Fixed processing of 422 clips.
#---------------
# Fixed processing of Y clips.
#---------------
# Initial version.


Function NNEDI3CL_rpow2(clip input, int "rfactor", int "field", int "nsize", int "nns", int "qual", int "etype", int "pscrn", int "device", bool "st", string "cshift", string "chroma_loc", bool "luma")
{
    rfactor = Default(rfactor, 2)
    field = Default(field, 1)
    step = Log(rfactor) / Log(2)
    Assert(Frac(step) == 0.0, "NNEDI3CL_rpow2: rfactor must be a power of two.")
    step = Int(step) - 1

    input
    for (i = 0, step)
    {
        NNEDI3CL(field=field, dh=true, dw=true, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn, device=device, st=st, luma=luma)
    }

    if (Defined(cshift))
    {
        chroma_loc = Default(chroma_loc, "left")

        Assert(cshift == "spline36" || cshift == "point" || cshift == "bilinear" || cshift == "bicubic" || cshift == "lanczos" || cshift == "lanczos4" || cshift == "spline16" || cshift == "spline64", "NNEDI3CL_rpow2: wrong cshift.")
        Assert(chroma_loc == "left" || chroma_loc == "center", "NNEDI3CL_rpow2: chroma_loc must be either left or center.")

        if (field == 1)
        {
            shift = -0.5 * (rfactor - 1)

            if (Is420() || Is422())
            {
                shift_chroma = ((chroma_loc == "left") ? shift : (-(rfactor - 1))) / 2.0
                v = (Is420()) ? BitRShiftU(Height(), 1) : Height()

                y = Eval("z_" + cshift + "Resize(ExtractY(), Width(), Height(), src_left=shift, src_top=shift)")
                u = Eval("z_" + cshift + "Resize(ExtractU(), BitRShiftU(Width(), 1), v, src_left=shift_chroma, src_top=shift)")
                v = Eval("z_" + cshift + "Resize(ExtractV(), BitRShiftU(Width(), 1), v, src_left=shift_chroma, src_top=shift)")

                return CombinePlanes(y, u, v, planes="yuv", sample_clip=last)
            }
            else
            {
                return Eval("z_" + cshift + "Resize(last, Width(), Height(), src_left=shift, src_top=shift)")
            }
        }
        else
        {
            shift = 0.5 * (rfactor - 1)
            shift_chroma =  (chroma_loc == "left") ? (0.5 * (rfactor * (step + 1) + 1) / 2.0) : shift
            v = (Is420()) ? BitRShiftU(Height(), 1) : Height()

            if (Is420() || Is422())
            {
                y = Eval("z_" + cshift + "Resize(ExtractY(), Width(), Height(), src_left=shift, src_top=shift)")
                u = Eval("z_" + cshift + "Resize(ExtractU(), BitRShiftU(Width(), 1), v, src_left=shift_chroma, src_top=shift)")
                v = Eval("z_" + cshift + "Resize(ExtractV(), BitRShiftU(Width(), 1), v, src_left=shift_chroma, src_top=shift)")

                return CombinePlanes(y, u, v, planes="yuv", sample_clip=last)
            }
            else
            {
                return Eval("z_" + cshift + "Resize(last, Width(), Height(), src_left=shift, src_top=shift)")
            }
        }
    }
}
