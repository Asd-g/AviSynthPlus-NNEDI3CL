#include <cstdio>
#include <cerrno>

#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_USE_OFFLINE_CACHE
#include "boost/compute/core.hpp"
#include "boost/dll.hpp"
#include "boost/compute/utility/dim.hpp"

#include "avisynth_c.h"
#include "NNEDI3CL.cl"

static constexpr int numNSIZE{ 7 };
static constexpr int numNNS{ 5 };
static constexpr int xdiaTable[numNSIZE]{ 8, 16, 32, 48, 8, 16, 32 };
static constexpr int ydiaTable[numNSIZE]{ 6, 6, 6, 6, 4, 4, 4 };
static constexpr int nnsTable[numNNS]{ 16, 32, 64, 128, 256 };

static std::mutex mtx;

struct NNEDI3CLData
{
    AVS_FilterInfo* fi;
    int field;
    int dh;
    int dw;
    bool process[3];
    int list_device;
    int info;
    boost::compute::command_queue queue;
    boost::compute::kernel kernel;
    boost::compute::image2d src;
    boost::compute::image2d dst;
    boost::compute::image2d tmp;
    boost::compute::buffer weights0;
    boost::compute::buffer weights1Buffer;
    cl_mem weights1;
    std::unique_ptr<char[]> err;

    void (*filter)(const AVS_VideoFrame* src, AVS_VideoFrame* dst, const int field_n, const NNEDI3CLData* const __restrict d);
};

static AVS_FORCEINLINE int roundds(const double f) noexcept
{
    return (f - std::floor(f) >= 0.5) ? std::min(static_cast<int>(std::ceil(f)), 32767) : std::max(static_cast<int>(std::floor(f)), -32768);
}

template<typename T, bool st>
void filter(const AVS_VideoFrame* src, AVS_VideoFrame* dst, const int field_n, const NNEDI3CLData* const __restrict d)
{
    const int planes_y[3]{ AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
    const int planes_r[3]{ AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B };
    const int* planes{ (avs_is_rgb(&d->fi->vi) ? planes_r : planes_y) };
    const int num_planes{ std::min(avs_num_components(&d->fi->vi), 3) };

    for (int i{ 0 }; i < num_planes; ++i)
    {
        if (d->process[i])
        {
            const int src_width{ static_cast<int>(avs_get_row_size_p(src, planes[i]) / sizeof(T)) };
            const int dst_width{ static_cast<int>(avs_get_row_size_p(dst, planes[i]) / sizeof(T)) };
            const int src_height{ avs_get_height_p(src, planes[i]) };
            const int dst_height{ avs_get_height_p(dst, planes[i]) };
            const T* srcp{ reinterpret_cast<const T*>(avs_get_read_ptr_p(src, planes[i])) };
            T* __restrict dstp{ reinterpret_cast<T*>(avs_get_write_ptr_p(dst, planes[i])) };

            auto queue{ d->queue };
            auto kernel{ d->kernel };
            auto src_image{ d->src };
            auto dst_image{ d->dst };
            auto tmp_image{ d->tmp };

            constexpr size_t localWorkSize[2]{ 4, 16 };

            queue.enqueue_write_image(src_image, boost::compute::dim(0, 0), boost::compute::dim(src_width, src_height), srcp, avs_get_pitch_p(src, planes[i]));

            if (d->dh && d->dw)
            {
                size_t globalWorkSize[]{ static_cast<size_t>(((src_height + 7) / 8 + 3) & -4), static_cast<size_t>((dst_width / 2 + 15) & -16) };
                kernel.set_args(src_image, tmp_image, d->weights0, d->weights1, src_height, src_width, src_height, dst_width, field_n, 1 - field_n, -1);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);

                globalWorkSize[0] = static_cast<size_t>(((dst_width + 7) / 8 + 3) & -4);
                globalWorkSize[1] = static_cast<size_t>((dst_height / 2 + 15) & -16);
                kernel.set_args(tmp_image, dst_image, d->weights0, d->weights1, dst_width, src_height, dst_width, dst_height, field_n, 1 - field_n, 0);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }
            else if (d->dw)
            {
                const size_t globalWorkSize[]{ static_cast<size_t>(((dst_height + 7) / 8 + 3) & -4), static_cast<size_t>((dst_width / 2 + 15) & -16) };
                kernel.set_args(src_image, dst_image, d->weights0, d->weights1, src_height, src_width, dst_height, dst_width, field_n, 1 - field_n, -1);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }
            else
            {
                const size_t globalWorkSize[]{ static_cast<size_t>(((dst_width + 7) / 8 + 3) & -4), static_cast<size_t>((dst_height / 2 + 15) & -16) };
                kernel.set_args(src_image, dst_image, d->weights0, d->weights1, src_width, src_height, dst_width, dst_height, field_n, 1 - field_n, 0);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }

            if constexpr (st)
            {
                std::lock_guard<std::mutex> lck(mtx);
                queue.enqueue_read_image(dst_image, boost::compute::dim(0, 0), boost::compute::dim(dst_width, dst_height), dstp, avs_get_pitch_p(dst, planes[i]));
            }
            else
                queue.enqueue_read_image(dst_image, boost::compute::dim(0, 0), boost::compute::dim(dst_width, dst_height), dstp, avs_get_pitch_p(dst, planes[i]));
        }
    }
}

/* multiplies and divides a rational number, such as a frame duration, in place and reduces the result */
AVS_FORCEINLINE void muldivRational(int64_t* num, int64_t* den, int64_t mul, int64_t div)
{
    /* do nothing if the rational number is invalid */
    if (!*den)
        return;

    int64_t a;
    int64_t b;
    *num *= mul;
    *den *= div;
    a = *num;
    b = *den;

    while (b != 0)
    {
        int64_t t{ a };
        a = b;
        b = t % b;
    }

    if (a < 0)
        a = -a;

    *num /= a;
    *den /= a;
}

AVS_VideoFrame* AVSC_CC NNEDI3CL_get_frame(AVS_FilterInfo* fi, int n)
{
    NNEDI3CLData* d{ static_cast<NNEDI3CLData*>(fi->user_data) };

    AVS_VideoFrame* src{ avs_get_frame(fi->child, (d->field > 1) ? (n / 2) : n) };
    AVS_VideoFrame* dst{ avs_new_video_frame_p(fi->env, &fi->vi, src) };

    int field{ d->field };
    if (field > 1)
        field -= 2;

    int err;
    const int64_t field_based{ avs_prop_get_int(fi->env, avs_get_frame_props_ro(fi->env, src), "_FieldBased", 0, &err) };
    if (err == 0)
    {
        if (field_based == 1)
            field = 0;
        else if (field_based == 2)
            field = 1;
    }

    int field_n;
    if (d->field > 1)
    {
        if (n & 1)
            field_n = (field == 0);
        else
            field_n = (field == 1);
    }
    else
        field_n = field;

    try
    {
        d->filter(src, dst, field_n, d);
    }
    catch (const boost::compute::opencl_error& error)
    {
        const std::string err{ std::string("NNEDI3CL: ") + error.error_string() };
        d->err = std::make_unique<char[]>(err.size() + 1);
        strcpy(d->err.get(), err.c_str());
        fi->error = d->err.get();
        avs_release_video_frame(src);
        avs_release_video_frame(dst);

        return nullptr;
    }

    AVS_Map* props{ avs_get_frame_props_rw(fi->env, dst) };
    avs_prop_set_int(fi->env, props, "_FieldBased", 0, 0);

    if (d->field > 1)
    {
        int errNum;
        int errDen;
        int64_t durationNum{ avs_prop_get_int(fi->env, props, "_DurationNum", 0, &errNum) };
        int64_t durationDen{ avs_prop_get_int(fi->env, props, "_DurationDen", 0, &errDen) };
        if (errNum == 0 && errDen == 0)
        {
            muldivRational(&durationNum, &durationDen, 1, 2);
            avs_prop_set_int(fi->env, props, "_DurationNum", durationNum, 0);
            avs_prop_set_int(fi->env, props, "_DurationDen", durationDen, 0);
        }
    }

    avs_release_video_frame(src);

    return dst;
}

void AVSC_CC free_NNEDI3CL(AVS_FilterInfo* fi)
{
    NNEDI3CLData* d{ static_cast<NNEDI3CLData*>(fi->user_data) };
    clReleaseMemObject(d->weights1);
    delete d;
}

int AVSC_CC NNEDI3CL_set_cache_hints(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}

AVS_Value AVSC_CC Create_NNEDI3CL(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum { Clip, Field, Dh, Dw, Planes, Nsize, Nns, Qual, Etype, Pscrn, Device, List_device, Info, St };

    NNEDI3CLData* params{ new NNEDI3CLData() };

    AVS_Clip* clip{ avs_new_c_filter(env, &params->fi, avs_array_elt(args, Clip), 1) };
    const AVS_VideoInfo& vi_temp{ params->fi->vi };
    AVS_Value v{ avs_void };

    try
    {
        if (!avs_is_planar(&params->fi->vi))
            throw std::string{ "only planar format is supported" };

        params->field = avs_as_int(avs_array_elt(args, Field));
        params->dh = avs_defined(avs_array_elt(args, Dh)) ? avs_as_bool(avs_array_elt(args, Dh)) : 0;
        params->dw = avs_defined(avs_array_elt(args, Dw)) ? avs_as_bool(avs_array_elt(args, Dw)) : 0;

        const int num_planes{ (avs_defined(avs_array_elt(args, Planes))) ? avs_array_size(avs_array_elt(args, Planes)) : 0 };

        for (int i{ 0 }; i < 3; ++i)
            params->process[i] = (num_planes <= 0);

        for (int i{ 0 }; i < num_planes; ++i)
        {
            const int n{ avs_as_int(*(avs_as_array(avs_array_elt(args, Planes)) + i)) };

            if (n >= avs_num_components(&params->fi->vi))
                throw std::string{ "plane index out of range" };

            if (params->process[n])
                throw std::string{ "plane specified twice" };

            params->process[n] = true;
        }

        const int nsize{ avs_defined(avs_array_elt(args, Nsize)) ? avs_as_int(avs_array_elt(args, Nsize)) : 6 };
        const int nns{ avs_defined(avs_array_elt(args, Nns)) ? avs_as_int(avs_array_elt(args, Nns)) : 1 };
        const int qual{ avs_defined(avs_array_elt(args, Qual)) ? avs_as_int(avs_array_elt(args, Qual)) : 1 };
        const int etype{ avs_defined(avs_array_elt(args, Etype)) ? avs_as_int(avs_array_elt(args, Etype)) : 0 };
        const int pscrn{ avs_defined(avs_array_elt(args, Pscrn)) ? avs_as_int(avs_array_elt(args, Pscrn)) : (avs_component_size(&params->fi->vi) < 4) ? 2 : 1 };
        const int device_id{ avs_defined(avs_array_elt(args, Device)) ? avs_as_int(avs_array_elt(args, Device)) : -1 };
        params->list_device = avs_defined(avs_array_elt(args, List_device)) ? avs_as_bool(avs_array_elt(args, List_device)) : 0;
        params->info = avs_defined(avs_array_elt(args, Info)) ? avs_as_bool(avs_array_elt(args, Info)) : 0;

        if (params->field < 0 || params->field > 3)
            throw std::string{ "field must be 0, 1, 2 or 3" };
        if (!params->dh && (params->fi->vi.height & 1))
            throw std::string{ "height must be mod 2 when dh=False" };
        if (params->dh && params->field > 1)
            throw std::string{ "field must be 0 or 1 when dh=True" };
        if (params->dw && params->field > 1)
            throw std::string{ "field must be 0 or 1 when dw=True" };
        if (nsize < 0 || nsize > 6)
            throw std::string{ "nsize must be 0, 1, 2, 3, 4, 5 or 6" };
        if (nns < 0 || nns > 4)
            throw std::string{ "nns must be 0, 1, 2, 3 or 4" };
        if (qual < 1 || qual > 2)
            throw std::string{ "qual must be 1 or 2" };
        if (etype < 0 || etype > 1)
            throw std::string{ "etype must be 0 or 1" };

        if (avs_component_size(&params->fi->vi) < 4)
        {
            if (pscrn < 1 || pscrn > 2)
                throw std::string{ "pscrn must be 1 or 2" };
        }
        else
        {
            if (pscrn != 1)
                throw std::string{ "pscrn must be 1 for float input" };
        }

        if (device_id >= static_cast<int>(boost::compute::system::device_count()))
            throw std::string{ "device index out of range" };

        if (params->list_device)
        {
            const auto devices{ boost::compute::system::devices() };
            std::string text;

            for (size_t i{ 0 }; i < devices.size(); ++i)
                text += std::to_string(i) + ": " + devices[i].name() + " (" + devices[i].platform().name() + ")" + "\n";

            params->err = std::make_unique<char[]>(text.size() + 1);
            strcpy(params->err.get(), text.c_str());

            AVS_Value cl{ avs_new_value_clip(clip) };
            AVS_Value args_[2]{ cl , avs_new_value_string(params->err.get()) };
            AVS_Value inv{ avs_invoke(params->fi->env, "Text", avs_new_value_array(args_, 2), 0) };
            AVS_Clip* clip1{ avs_take_clip(inv, env) };

            v = avs_new_value_clip(clip1);

            avs_release_clip(clip1);
            avs_release_value(inv);
            avs_release_value(cl);
            avs_release_clip(clip);

            return v;
        }

        boost::compute::device device{ boost::compute::system::default_device() };

        if (device_id > -1)
            device = boost::compute::system::devices().at(device_id);

        boost::compute::context context{ device };
        params->queue = boost::compute::command_queue{ context, device };

        if (params->info)
        {
            std::string text{ "=== Platform Info ===\n" };
            const auto platform{ device.platform() };
            text += "Profile: " + platform.get_info<CL_PLATFORM_PROFILE>() + "\n";
            text += "Version: " + platform.get_info<CL_PLATFORM_VERSION>() + "\n";
            text += "Name: " + platform.get_info<CL_PLATFORM_NAME>() + "\n";
            text += "Vendor: " + platform.get_info<CL_PLATFORM_VENDOR>() + "\n";

            text += "\n";

            text += "=== Device Info ===\n";
            text += "Name: " + device.get_info<CL_DEVICE_NAME>() + "\n";
            text += "Vendor: " + device.get_info<CL_DEVICE_VENDOR>() + "\n";
            text += "Profile: " + device.get_info<CL_DEVICE_PROFILE>() + "\n";
            text += "Version: " + device.get_info<CL_DEVICE_VERSION>() + "\n";
            text += "Max compute units: " + std::to_string(device.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>()) + "\n";
            text += "Max work-group size: " + std::to_string(device.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
            const auto max_work_item_sizes{ device.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>() };
            text += "Max work-item sizes: " + std::to_string(max_work_item_sizes[0]) + ", " + std::to_string(max_work_item_sizes[1]) + ", " + std::to_string(max_work_item_sizes[2]) + "\n";
            text += "2D image max width: " + std::to_string(device.get_info<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) + "\n";
            text += "2D image max height: " + std::to_string(device.get_info<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) + "\n";
            text += "Image support: " + std::string{ device.get_info<CL_DEVICE_IMAGE_SUPPORT>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            const auto global_mem_cache_type{ device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() };
            if (global_mem_cache_type == CL_NONE)
                text += "Global memory cache type: CL_NONE\n";
            else if (global_mem_cache_type == CL_READ_ONLY_CACHE)
                text += "Global memory cache type: CL_READ_ONLY_CACHE\n";
            else if (global_mem_cache_type == CL_READ_WRITE_CACHE)
                text += "Global memory cache type: CL_READ_WRITE_CACHE\n";
            text += "Global memory cache size: " + std::to_string(device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() / 1024) + " KB\n";
            text += "Global memory size: " + std::to_string(device.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)) + " MB\n";
            text += "Max constant buffer size: " + std::to_string(device.get_info<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() / 1024) + " KB\n";
            text += "Max constant arguments: " + std::to_string(device.get_info<CL_DEVICE_MAX_CONSTANT_ARGS>()) + "\n";
            text += "Local memory type: " + std::string{ device.get_info<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL" } + "\n";
            text += "Local memory size: " + std::to_string(device.get_info<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024) + " KB\n";
            text += "Available: " + std::string{ device.get_info<CL_DEVICE_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            text += "Compiler available: " + std::string{ device.get_info<CL_DEVICE_COMPILER_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            text += "OpenCL C version: " + device.get_info<CL_DEVICE_OPENCL_C_VERSION>() + "\n";
            text += "Linker available: " + std::string{ device.get_info<CL_DEVICE_LINKER_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            text += "Image max buffer size: " + std::to_string(device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) / 1024) + " KB" + "\n";
            text += "Out of order (on host): " + std::string{ !!(device.get_info<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>() & 1) ? "CL_TRUE" : "CL_FALSE" } + "\n";
            text += "Out of order (on device): " + std::string{ !!(device.get_info<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>() & 1) ? "CL_TRUE" : "CL_FALSE" };

            params->err = std::make_unique<char[]>(text.size() + 1);
            strcpy(params->err.get(), text.c_str());

            AVS_Value cl{ avs_new_value_clip(clip) };
            AVS_Value args_[2]{ cl, avs_new_value_string(params->err.get()) };
            AVS_Value inv{ avs_invoke(params->fi->env, "Text", avs_new_value_array(args_, 2), 0) };
            AVS_Clip* clip1{ avs_take_clip(inv, env) };

            v = avs_new_value_clip(clip1);

            avs_release_clip(clip1);
            avs_release_value(inv);
            avs_release_value(cl);
            avs_release_clip(clip);

            return v;
        }

        if (params->field > 1)
        {
            if (params->fi->vi.num_frames > INT_MAX / 2)
                throw std::string{ "resulting clip is too long" };

            params->fi->vi.num_frames *= 2;

            int64_t fps_n{ params->fi->vi.fps_numerator };
            int64_t fps_d{ params->fi->vi.fps_denominator };
            muldivRational(&fps_n, &fps_d, 2, 1);
            params->fi->vi.fps_numerator = static_cast<unsigned>(fps_n);
            params->fi->vi.fps_denominator = static_cast<unsigned>(fps_d);
        }

        if (params->dh)
            params->fi->vi.height *= 2;

        if (params->dw)
            params->fi->vi.width *= 2;

        const int peak{ (1 << avs_bits_per_component(&params->fi->vi)) - 1 };

        std::string weightsPath{ boost::dll::this_line_location().parent_path().generic_string() + "/nnedi3_weights.bin" };

        FILE* weightsFile{ nullptr };
#ifdef _WIN32
        const int requiredSize{ MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, nullptr, 0) };
        std::unique_ptr<wchar_t[]> wbuffer{ std::make_unique<wchar_t[]>(requiredSize) };
        MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, wbuffer.get(), requiredSize);
        weightsFile = _wfopen(wbuffer.get(), L"rb");
#else
        weightsFile = std::fopen(weightsPath.c_str(), "rb");
#endif

#if !defined(_WIN32) && defined(NNEDI3_DATADIR)
        if (!weightsFile)
        {
            weightsPath = std::string{ NNEDI3_DATADIR } + "/nnedi3_weights.bin";
            weightsFile = std::fopen(weightsPath.c_str(), "rb");
        }
#endif
        if (!weightsFile)
            throw std::string{ "error opening file " + weightsPath + " (" + std::strerror(errno) + ")" };

        if (std::fseek(weightsFile, 0, SEEK_END))
        {
            std::fclose(weightsFile);
            throw std::string{ "error seeking to the end of file " + weightsPath + " (" + std::strerror(errno) + ")" };
        }

        constexpr long correctSize{ 13574928 }; // Version 0.9.4 of the Avisynth plugin
        const long weightsSize{ std::ftell(weightsFile) };

        if (weightsSize == -1)
        {
            std::fclose(weightsFile);
            throw std::string{ "error determining the size of file " + weightsPath + " (" + std::strerror(errno) + ")" };
        }
        else if (weightsSize != correctSize)
        {
            std::fclose(weightsFile);
            throw std::string{ "incorrect size of file " + weightsPath + ". Should be " + std::to_string(correctSize) + " bytes, but got " + std::to_string(weightsSize) + " bytes instead" };
        }

        std::rewind(weightsFile);

        float* bdata{ reinterpret_cast<float*>(malloc(correctSize)) };
        const size_t bytesRead{ std::fread(bdata, 1, correctSize, weightsFile) };

        if (bytesRead != correctSize)
        {
            std::fclose(weightsFile);
            free(bdata);
            throw std::string{ "error reading file " + weightsPath + ". Should read " + std::to_string(correctSize) + " bytes, but read " + std::to_string(bytesRead) + " bytes instead" };
        }

        std::fclose(weightsFile);

        constexpr int dims0{ 49 * 4 + 5 * 4 + 9 * 4 };
        constexpr int dims0new{ 4 * 65 + 4 * 5 };
        const int dims1{ nnsTable[nns] * 2 * (xdiaTable[nsize] * ydiaTable[nsize] + 1) };
        int dims1tsize{ 0 };
        int dims1offset{ 0 };

        for (int j{ 0 }; j < numNNS; ++j)
        {
            for (int i{ 0 }; i < numNSIZE; ++i)
            {
                if (i == nsize && j == nns)
                    dims1offset = dims1tsize;

                dims1tsize += nnsTable[j] * 2 * (xdiaTable[i] * ydiaTable[i] + 1) * 2;
            }
        }

        float* weights0{ new float[std::max(dims0, dims0new)] };
        float* weights1{ new float[dims1 * 2] };

        // Adjust prescreener weights
        if (pscrn == 2) // using new prescreener
        {
            int* offt{ reinterpret_cast<int*>(calloc(4 * 64, sizeof(int))) };

            for (int j{ 0 }; j < 4; ++j)
            {
                for (int k{ 0 }; k < 64; ++k)
                    offt[j * 64 + k] = ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
            }

            const float* bdw{ bdata + dims0 + dims0new * (pscrn - 2) };
            short* ws{ reinterpret_cast<short*>(weights0) };
            float* wf{ reinterpret_cast<float*>(&ws[4 * 64]) };
            double mean[4]{ 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j{ 0 }; j < 4; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < 64; ++k)
                    cmean += bdw[offt[j * 64 + k]];

                mean[j] = cmean / 64.0;
            }

            const double half{ peak / 2.0 };

            // Factor mean removal and 1.0/half scaling into first layer weights. scale to int16 range
            for (int j{ 0 }; j < 4; ++j)
            {
                double mval{ 0.0 };
                for (int k{ 0 }; k < 64; ++k)
                    mval = std::max(mval, std::abs((bdw[offt[j * 64 + k]] - mean[j]) / half));

                const double scale{ 32767.0 / mval };

                for (int k{ 0 }; k < 64; ++k)
                    ws[offt[j * 64 + k]] = roundds(((bdw[offt[j * 64 + k]] - mean[j]) / half) * scale);

                wf[j] = static_cast<float>(mval / 32767.0);
            }

            memcpy(wf + 4, bdw + 4 * 64, (dims0new - 4 * 64) * sizeof(float));
            free(offt);
        }
        else // using old prescreener
        {
            double mean[4]{ 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j{ 0 }; j < 4; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < 48; ++k)
                    cmean += bdata[j * 48 + k];

                mean[j] = cmean / 48.0;
            }

            const double half{ ((avs_component_size(&params->fi->vi) < 4) ? peak : 1.0) / 2.0 };

            // Factor mean removal and 1.0/half scaling into first layer weights
            for (int j{ 0 }; j < 4; ++j)
            {
                for (int k{ 0 }; k < 48; ++k)
                    weights0[j * 48 + k] = static_cast<float>((bdata[j * 48 + k] - mean[j]) / half);
            }

            memcpy(weights0 + 4 * 48, bdata + 4 * 48, (dims0 - 4 * 48) * sizeof(float));
        }

        // Adjust prediction weights
        for (int i{ 0 }; i < 2; ++i)
        {
            const float* bdataT{ bdata + dims0 + dims0new * 3 + dims1tsize * etype + dims1offset + i * dims1 };
            float* weightsT{ weights1 + i * dims1 };
            const int nnst{ nnsTable[nns] };
            const int asize{ xdiaTable[nsize] * ydiaTable[nsize] };
            const int boff{ nnst * 2 * asize };
            double* mean{ reinterpret_cast<double*>(calloc(asize + 1 + nnst * 2, sizeof(double))) };

            // Calculate mean weight of each neuron (ignore bias)
            for (int j{ 0 }; j < nnst * 2; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < asize; ++k)
                    cmean += bdataT[j * asize + k];

                mean[asize + 1 + j] = cmean / asize;
            }

            // Calculate mean softmax neuron
            for (int j{ 0 }; j < nnst; ++j)
            {
                for (int k{ 0 }; k < asize; ++k)
                    mean[k] += bdataT[j * asize + k] - mean[asize + 1 + j];

                mean[asize] += bdataT[boff + j];
            }

            for (int j{ 0 }; j < asize + 1; ++j)
                mean[j] /= nnst;

            // Factor mean removal into weights, and remove global offset from softmax neurons
            for (int j{ 0 }; j < nnst * 2; ++j)
            {
                for (int k{ 0 }; k < asize; ++k)
                {
                    const double q{ (j < nnst) ? mean[k] : 0.0 };
                    weightsT[j * asize + k] = static_cast<float>(bdataT[j * asize + k] - mean[asize + 1 + j] - q);
                }

                weightsT[boff + j] = static_cast<float>(bdataT[boff + j] - (j < nnst ? mean[asize] : 0.0));
            }

            free(mean);
        }

        free(bdata);

        const int xdia{ xdiaTable[nsize] };
        const int ydia{ ydiaTable[nsize] };
        const int asize{ xdiaTable[nsize] * ydiaTable[nsize] };
        const int xdiad2m1{ std::max(xdia, (pscrn == 1) ? 12 : 16) / 2 - 1 };
        const int ydiad2m1{ ydia / 2 - 1 };
        const int xOffset{ (xdia == 8) ? (pscrn == 1 ? 2 : 4) : 0 };
        const int inputWidth{ std::max(xdia, (pscrn == 1) ? 12 : 16) + 32 - 1 };
        const int inputHeight{ ydia + 16 - 1 };
        const float scaleAsize{ 1.0f / asize };
        const float scaleQual{ 1.0f / qual };

        params->weights0 = boost::compute::buffer{ context, std::max(dims0, dims0new) * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weights0 };
        params->weights1Buffer = boost::compute::buffer{ context, dims1 * 2 * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weights1 };
        delete[] weights0;
        delete[] weights1;

        if (static_cast<size_t>(dims1 * 2) > device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE))
            throw std::string{ "the device's image max buffer size is too small. Reduce nsize/nns...or buy a new graphics card" };

        boost::compute::program program;
        try
        {
            std::ostringstream options;
            options.imbue(std::locale{ "C" });
            options.precision(16);
            options.setf(std::ios::fixed, std::ios::floatfield);
            options << "-cl-denorms-are-zero -cl-fast-relaxed-math -Werror";
            options << " -D QUAL=" << qual;
            if (pscrn == 1)
            {
                options << " -D PRESCREEN=prescreenOld";
                options << " -D USE_OLD_PSCRN=1";
                options << " -D USE_NEW_PSCRN=0";
            }
            else
            {
                options << " -D PRESCREEN=prescreenNew";
                options << " -D USE_OLD_PSCRN=0";
                options << " -D USE_NEW_PSCRN=1";
            }
            options << " -D PSCRN_OFFSET=" << (pscrn == 1 ? 5 : 6);
            options << " -D DIMS1=" << dims1;
            options << " -D NNS=" << nnsTable[nns];
            options << " -D NNS2=" << (nnsTable[nns] * 2);
            options << " -D XDIA=" << xdia;
            options << " -D YDIA=" << ydia;
            options << " -D ASIZE=" << asize;
            options << " -D XDIAD2M1=" << xdiad2m1;
            options << " -D YDIAD2M1=" << ydiad2m1;
            options << " -D X_OFFSET=" << xOffset;
            options << " -D INPUT_WIDTH=" << inputWidth;
            options << " -D INPUT_HEIGHT=" << inputHeight;
            options << " -D SCALE_ASIZE=" << scaleAsize << "f";
            options << " -D SCALE_QUAL=" << scaleQual << "f";
            options << " -D PEAK=" << peak;
            if (!(params->dh || params->dw))
            {
                options << " -D Y_OFFSET=" << (ydia - 1);
                options << " -D Y_STEP=2";
                options << " -D Y_STRIDE=32";
            }
            else
            {
                options << " -D Y_OFFSET=" << (ydia / 2);
                options << " -D Y_STEP=1";
                options << " -D Y_STRIDE=16";
            }

            program = boost::compute::program::build_with_source(source, context, options.str());
        }
        catch (const boost::compute::opencl_error& error)
        {
            throw error.error_string() + "\n" + program.build_log();
        }

        if (avs_component_size(&params->fi->vi) < 4)
            params->kernel = program.create_kernel("filter_uint");
        else
            params->kernel = program.create_kernel("filter_float");

        const int st{ avs_defined(avs_array_elt(args, St)) ? avs_as_bool(avs_array_elt(args, St)) : !!(device.get_info<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>() & 1) };
        cl_image_format imageFormat;

        switch (avs_component_size(&params->fi->vi))
        {
            case 1:
            {
                imageFormat = { CL_R, CL_UNSIGNED_INT8 };
                params->filter = (st) ? filter<uint8_t, true> : filter<uint8_t, false>;
                break;
            }
            case 2:
            {
                imageFormat = { CL_R, CL_UNSIGNED_INT16 };
                params->filter = (st) ? filter<uint16_t, true> : filter<uint16_t, false>;
                break;
            }
            default:
            {
                imageFormat = { CL_R, CL_FLOAT };
                params->filter = (st) ? filter<float, true> : filter<float, false>;
            }
        }

        params->src = boost::compute::image2d{ context,
                                   static_cast<size_t>(vi_temp.width),
                                   static_cast<size_t>(vi_temp.height),
                                   boost::compute::image_format{ imageFormat },
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };

        params->dst = boost::compute::image2d{ context,
                                   static_cast<size_t>(std::max(params->fi->vi.width, params->fi->vi.height)),
                                   static_cast<size_t>(std::max(params->fi->vi.width, params->fi->vi.height)),
                                   boost::compute::image_format{ imageFormat },
                                   CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY };

        params->tmp = (params->dh && params->dw) ? boost::compute::image2d{ context,
                                                      static_cast<size_t>(std::max(params->fi->vi.width, params->fi->vi.height)),
                                                      static_cast<size_t>(std::max(params->fi->vi.width, params->fi->vi.height)),
                                                      boost::compute::image_format{ imageFormat },
                                                      CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS }
        : boost::compute::image2d{};

        {
            constexpr cl_image_format format{ CL_R, CL_FLOAT };

            cl_image_desc desc;
            desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width = dims1 * 2;
            desc.image_height = 1;
            desc.image_depth = 1;
            desc.image_array_size = 0;
            desc.image_row_pitch = 0;
            desc.image_slice_pitch = 0;
            desc.num_mip_levels = 0;
            desc.num_samples = 0;
#ifdef BOOST_COMPUTE_CL_VERSION_2_0
            desc.mem_object = params->weights1Buffer.get();
#else
            desc.buffer = d->weights1Buffer.get();
#endif

            cl_int error{ 0 };

            cl_mem mem{ clCreateImage(context, 0, &format, &desc, nullptr, &error) };
            if (!mem)
                BOOST_THROW_EXCEPTION(boost::compute::opencl_error(error));

            params->weights1 = mem;
        }
    }
    catch (const std::string& error)
    {
        const std::string err{ std::string("NNEDI3CL: ") + error };
        params->err = std::make_unique<char[]>(err.size() + 1);
        strcpy(params->err.get(), err.c_str());
        v = avs_new_value_error(params->err.get());
    }
    catch (const boost::compute::no_device_found& error)
    {
        const std::string err{ std::string{ "NNEDI3CL: " } + error.what() };
        params->err = std::make_unique<char[]>(err.size() + 1);
        strcpy(params->err.get(), err.c_str());
        v = avs_new_value_error(params->err.get());
    }
    catch (const boost::compute::opencl_error& error)
    {
        const std::string err{ std::string{ "NNEDI3CL: " } + error.error_string() };
        params->err = std::make_unique<char[]>(err.size() + 1);
        strcpy(params->err.get(), err.c_str());
        v = avs_new_value_error(params->err.get());
    }

    if (!avs_defined(v))
    {
        v = avs_new_value_clip(clip);

        params->fi->user_data = reinterpret_cast<void*>(params);
        params->fi->get_frame = NNEDI3CL_get_frame;
        params->fi->set_cache_hints = NNEDI3CL_set_cache_hints;
        params->fi->free_filter = free_NNEDI3CL;
    }

    avs_release_clip(clip);

    return v;
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "NNEDI3CL", "ci[dh]b[dw]b[planes]i*[nsize]i[nns]i[qual]i[etype]i[pscrn]i[device]i[list_device]b[info]b[st]b", Create_NNEDI3CL, 0);
    return "NNEDI3CL";
}
