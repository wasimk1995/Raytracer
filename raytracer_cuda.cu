// A very basic raytracer example.
// [compile]
// nvcc -o raytracer_cuda raytracer_cuda.cu
// [/compile]
// [ignore]
// Copyright (C) 2012  www.scratchapixel.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// [/ignore]
#include <cstdlib>
#include <cstdio>
#include "math.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <time.h>

using namespace std;


#define GIG 1000000000
#define CPG 2.9           // Cycles per GHz -- Adjust to your computer

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#define INFINITY 1e8
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#define PRINT_TIME 1

// This variable controls the maximum recursion depth
#define MAX_RAY_DEPTH 5 

template<typename T>
class Vec3
{
public:
    T x, y, z;
    CUDA_CALLABLE_MEMBER Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    CUDA_CALLABLE_MEMBER Vec3(T xx) : x(xx), y(xx), z(xx) {}
    CUDA_CALLABLE_MEMBER Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    CUDA_CALLABLE_MEMBER Vec3& normalize()
    {
        T nor2 = length2();
        if (nor2 > 0) {
            T invNor = 1 / sqrt(nor2);
            x *= invNor, y *= invNor, z *= invNor;
        }
        return *this;
    }
    CUDA_CALLABLE_MEMBER Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
    CUDA_CALLABLE_MEMBER Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
    CUDA_CALLABLE_MEMBER T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
    CUDA_CALLABLE_MEMBER Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
    CUDA_CALLABLE_MEMBER Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
    CUDA_CALLABLE_MEMBER Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
    CUDA_CALLABLE_MEMBER Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    CUDA_CALLABLE_MEMBER Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
    CUDA_CALLABLE_MEMBER T length2() const { return x * x + y * y + z * z; }
    CUDA_CALLABLE_MEMBER T length() const { return sqrt(length2()); }
    CUDA_CALLABLE_MEMBER friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    {
        os << "[" << v.x << " " << v.y << " " << v.z << "]";
        return os;
    }
};

typedef Vec3<float> Vec3f;

class Sphere
{
public:
    Vec3f center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
    float transparency, reflection;         /// surface transparency and reflectivity
    CUDA_CALLABLE_MEMBER Sphere(
        const Vec3f &c,             // consts here
        const float &r,
        const Vec3f &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vec3f &ec = 0) :
        center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
        transparency(transp), reflection(refl)
    { /* empty */ }
    // Compute a ray-sphere intersection using the geometric solution
    CUDA_CALLABLE_MEMBER bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
    {
        Vec3f l = center - rayorig;
        float tca = l.dot(raydir);
        if (tca < 0) return false;
        float d2 = l.dot(l) - tca * tca;
        if (d2 > radius2) return false;
        float thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        
        return true;
    }
};

//cpu time calculation
struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

double timeInSeconds(struct timespec* t)
{
    return (t->tv_sec + 1.0e-9 * (t->tv_nsec));
}

__host__ __device__ float mix(const float &a, const float &b, const float &mix)
{
    return b * mix + a * (1 - mix);
}


/**************************************************************************************************************************************************/
/***************************THIS IS THE CPU HOST VERSION***************************/
// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
__host__ __device__ Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    //const std::vector<Sphere> &spheres,
    const Sphere* spheres,
    const int &depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    float tnear = INFINITY;
    const Sphere* sphere = NULL;
    int size = 6;
    // find intersection of this ray with the sphere in the scene
    for (unsigned i = 0; i < size; ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                sphere = &spheres[i];
            }
        }
    }
    // if there's no intersection return black or background color
    if (!sphere) return Vec3f(2);
    Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
    Vec3f phit = rayorig + raydir * tnear; // point of intersection
    Vec3f nhit = phit - sphere->center; // normal at the intersection point
    nhit.normalize(); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
    //if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -raydir.dot(nhit);
        // change the mix value to tweak the effect
        float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
        refldir.normalize();
        Vec3f reflection = 0;// trace(phit + nhit * bias, refldir, spheres, depth + 1);
        Vec3f refraction = 0;
        // if the sphere is also transparent compute refraction ray (transmission)
        //if (sphere->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -nhit.dot(raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
            refrdir.normalize();
            //refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1);
        //}
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = (
            reflection * fresneleffect +
            refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
    //}
    //else {
        // it's a diffuse object, no need to raytrace any further
        for (unsigned i = 0; i < size; ++i) {
            if (spheres[i].emissionColor.x > 0) {
                // this is a light
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[i].center - phit;
                lightDirection.normalize();
                for (unsigned j = 0; j < size; ++j) {
                    if (i != j) {
                        float t0, t1;
                        if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = 0;
                            break;
                        }
                    }
                }
                surfaceColor += sphere->surfaceColor * transmission *
                max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
        }
    //}
    
    return surfaceColor + sphere->emissionColor;
}

// Main rendering function. We compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
void render(Sphere* spheres)
{
    unsigned width = 1920, height = 1080;
    Vec3f *image = new Vec3f[width * height], *pixel = image;
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectratio = width / float(height);
    float angle = tan(M_PI * 0.5 * fov / 180.);
    // Trace rays
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x, ++pixel) {
            float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
            float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
            Vec3f raydir(xx, yy, -1);
            raydir.normalize();
            *pixel = trace(Vec3f(0), raydir, spheres, 0);
        }
    }
    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::ofstream ofs("./cpu_untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i <width*height; i++) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
               (unsigned char)(std::min(float(1), image[i].y) * 255) <<
               (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
    delete [] image;
}


/**************************************************************************************************************************************************/
/*********************************CUDA PART BEGINS********************************/
//CUDA Trace Function first calculates all of the 63 rays with a max depth of 5.
//The information is stores are the ray origins and ray directions of each ray
//and also the sphere that it intersected with
__device__ Vec3f trace_cuda(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const Sphere *spheres, 
    Vec3f* rays_orig, 
    Vec3f* rays_dir, 
    Vec3f* surfaceColors, 
    const Sphere** inter_spheres,
    int k)
{
    int sphere_size = 6;
    float bias = 1e-4; // add some bias to the point from which we will be tracing

    for(int i = 0; i < 31; i++){
        float tnear = INFINITY;
        // find intersection of this ray with the sphere in the scene
        for (int j = 0; j < sphere_size; j++) {
            float t0 = INFINITY, t1 = INFINITY;
            if (spheres[j].intersect(rays_orig[i*k], rays_dir[i*k], t0, t1)) {
                if (t0 < 0) t0 = t1;
                if (t0 < tnear) {
                    tnear = t0;
                    inter_spheres[i*k] = &spheres[j];
                }
                else
                    inter_spheres[i*k] = NULL;
            }
        }
        // if there's no intersection then the resulting rays will be the same ray
        if (!inter_spheres[i*k]){
            rays_orig[(i*2+1)*k] = rays_orig[i*k];
            rays_dir[(i*2+1)*k] = rays_dir[i*k];
            rays_orig[(i*2+2)*k] = rays_orig[i*k];
            rays_dir[(i*2+2)*k] = rays_dir[i*k];
            continue;
        }

        Vec3f phit = rays_orig[i*k] + rays_dir[i*k] * tnear; // point of intersection
        Vec3f nhit = phit - inter_spheres[i*k]->center; // normal at the intersection point
        nhit.normalize(); // normalize normal direction
        // If the normal and the view direction are not opposite to each other
        // reverse the normal direction. That also means we are inside the sphere so set
        // the inside bool to true. Finally reverse the sign of IdotN which we want
        // positive.
        bool inside = false;
        if (rays_dir[i*k].dot(nhit) > 0) nhit = -nhit, inside = true;

        if ((inter_spheres[i*k]->transparency > 0 || inter_spheres[i*k]->reflection > 0)){
            Vec3f refldir = rays_dir[i*k] - nhit * 2 * rays_dir[i*k].dot(nhit);
            refldir.normalize();
            rays_dir[(i*2+1)*k] = refldir;
            rays_orig[(i*2+1)*k] = phit + nhit * bias;

            if (inter_spheres[i*k]->transparency){
                float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
                float cosi = -nhit.dot(rays_dir[i*k]);
                float k = 1 - eta * eta * (1 - cosi * cosi);
                Vec3f refrdir = rays_dir[i*k] * eta + nhit * (eta *  cosi - sqrtf(k));
                refrdir.normalize();
                rays_dir[(i*2+2)*k] = refrdir;
                rays_orig[(i*2+2)*k] = phit - nhit * bias;
            }
        }
    }
    for(int i = 31; i < 63; i++){
        float tnear = INFINITY;
        // find intersection of this ray with the sphere in the scene
        for (unsigned j = 0; j < sphere_size; ++j) {
            float t0 = INFINITY, t1 = INFINITY;
            if (spheres[j].intersect(rays_orig[i], rays_dir[i], t0, t1)) {
                if (t0 < 0) t0 = t1;
                if (t0 < tnear) {
                    tnear = t0;
                    inter_spheres[i*k] = &spheres[j];
                }
                else
                    inter_spheres[i*k] = NULL;
            }
        }
    }

    //Go Backwards to find the surface color and return final color
    int start, end, depth;
    depth = 5;
    end = 63;
    start = end - (int)pow(2,depth) - 1;
    for(int i = depth; i >= 0; i--){
        for(int i = start; i < end; i--){
            Vec3f surfaceColor = 0;
            if(!inter_spheres[i*k]) surfaceColors[i*k] = Vec3f(2);
            else{
                // this is a light
                float tnear = INFINITY;
                float t0 = INFINITY, t1 = INFINITY;
                if (inter_spheres[i*k]->intersect(rays_orig[i*k], rays_dir[i*k], t0, t1)) {
                    if (t0 < 0) t0 = t1;
                    if (t0 < tnear) {
                        tnear = t0;
                        inter_spheres[i*k] = &spheres[j];
                    }
                    else
                        inter_spheres[i*k] = NULL;
                }
                Vec3f phit = rays_orig[i*k] + rays_dir[i*k] * tnear; // point of intersection
                Vec3f nhit = phit - inter_spheres[i*k]->center; // normal at the intersection point
                nhit.normalize(); // normalize normal direction
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[j].center - phit;
                lightDirection.normalize();
                for (int k = 0; k < sphere_size; k++) {
                    if (j != k) {
                        float t0, t1;
                        if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = 0;
                            break;
                        }
                    }
                }
                surfaceColor += inter_spheres[i]->surfaceColor * transmission *
                fmaxf(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
                }
                surfaceColors[i] = surfaceColor;
            }
        }
    }

    for(int i = 30; i >= 0; i--){
        Vec3f surfaceColor = 0;
        if(!inter_spheres[i]) surfaceColors[i] = Vec3f(2);
        else{
            float tnear = INFINITY;
            float t0 = INFINITY, t1 = INFINITY;
            if (inter_spheres[i]->intersect(rays_orig[i], rays_dir[i], t0, t1)) {
                if (t0 < 0) t0 = t1;
                if (t0 < tnear) {
                    tnear = t0;
                }
            }
            Vec3f phit = rays_orig[i] + rays_dir[i] * tnear; // point of intersection
            Vec3f nhit = phit - inter_spheres[i]->center; // normal at the intersection point
            nhit.normalize(); // normalize normal direction
            // If the normal and the view direction are not opposite to each other
            // reverse the normal direction. That also means we are inside the sphere so set
            // the inside bool to true. Finally reverse the sign of IdotN which we want
            // positive.
            if (rays_dir[i].dot(nhit) > 0) nhit = -nhit;

            float facingratio = -rays_dir[i].dot(nhit);
            // change the mix value to tweak the effect
            float fresneleffect = mix(powf((float)(1 - facingratio), 3.0), 1, 0.1);

            // the result is a mix of reflection and refraction (if the sphere is transparent)
            surfaceColors[i] = (
                surfaceColors[i*2+1] * fresneleffect +
                surfaceColors[i*2+2] * (1 - fresneleffect) * inter_spheres[i]->transparency) * inter_spheres[i]->surfaceColor;
        }
    }
    return surfaceColors[0];
}

//CUDA Rener Function
__global__ void render_cuda(Sphere *d_spheres, int height, int width, Vec3f *pixel, Vec3f* d_rays_orig, Vec3f* d_rays_dir, Vec3f* d_surfaceColors, Sphere** d_inter_spheres)
{
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectratio = width / float(height);
    float angle = tanf(M_PI * 0.5 * fov / 180.);

    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int k = y*width+x;
    float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
    float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
    Vec3f raydir(xx, yy, -1);
    raydir.normalize();
    if(k < height*width)
        pixel[k] = trace_cuda(Vec3f(0), raydir, d_spheres, d_rays_orig, d_rays_dir, d_surfaceColors, d_inter_spheres, k);
        //for(int i = 0; i < 63; i++)
            //trace(Vec3f(0),raydir,d_spheres,0);
}


/**************************************************************************************************************************************************/

// In the main function, we will create the scene which is composed of 5 spheres
// and 1 light (which is also a sphere). Then, once the scene description is complete
// we render that scene, by calling the render() function.
int main(int argc, char **argv)
{
    // dimensions of image
    int width = 7680;
    int height = 4320;
    int num_pixels = width*height;

    //CPU timing variables
    struct timespec t1, t2;
    float cpu_time;

    // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;

    srand48(13);
    std::vector<Sphere> spheres;
    // position, radius, surface color, reflectivity, transparency, emission color
    spheres.push_back(Sphere(Vec3f( 0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
    spheres.push_back(Sphere(Vec3f( 0.0,      0, -20),     4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
    spheres.push_back(Sphere(Vec3f( 5.0,     -1, -15),     2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
    spheres.push_back(Sphere(Vec3f( 5.0,      0, -25),     3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
    spheres.push_back(Sphere(Vec3f(-5.5,      0, -15),     3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));
    // light
    spheres.push_back(Sphere(Vec3f( 0.0,     20, -30),     3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));

    //CPU num times trace is run
    /*int *num;
    int numSize = width*height*sizeof(int);
    num = (int*)malloc(numSize);
    memset(num,0,numSize);*/
    //CPU answers

    /*clock_gettime(CLOCK_MONOTONIC,&t1);
    render(spheres);
    clock_gettime(CLOCK_MONOTONIC,&t2);
    cpu_time = timeInSeconds(&t2) - timeInSeconds(&t1);
    cout << "CPU time is: " << cpu_time << " (sec)" << endl;*/

    //CPU Arrays
    Vec3f *image;
    Sphere *h_spheres;
    Vec3f *h_rays_orig;
    Vec3f *h_rays_dir;
    Sphere **h_inter_spheres;
    Vec3f *h_surfaceColors;

    //Image Array
    int flatArraySize = width * height * sizeof(Vec3f);
    image = (Vec3f*)malloc(flatArraySize);

    //Sphere Array
    int num_spheres = 6;
    int size = num_spheres * sizeof(Sphere);
    h_spheres = (Sphere*)malloc(size);
    h_spheres[0] = Sphere(Vec3f( 0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0);
    h_spheres[1] = Sphere(Vec3f( 0.0,      0, -20),     4, Vec3f(1.00, 0.32, 0.36), 1, 0.5);
    h_spheres[2] = Sphere(Vec3f( 5.0,     -1, -15),     2, Vec3f(0.90, 0.76, 0.46), 1, 0.0);
    h_spheres[3] = Sphere(Vec3f( 5.0,      0, -25),     3, Vec3f(0.65, 0.77, 0.97), 1, 0.0);
    h_spheres[4] = Sphere(Vec3f(-5.5,      0, -15),     3, Vec3f(0.90, 0.90, 0.90), 1, 0.0);
    h_spheres[5] = Sphere(Vec3f( 0.0,     20, -30),     3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3));

    int vec_store_size = sizeof(Vec3f)*63*num_pixels;
    int sphere_ptr_size = sizeof(Sphere*)*63*num_pixels;

    h_rays_orig = (Vec3f*)malloc(vec_store_size);
    h_rays_dir = (Vec3f*)malloc(vec_store_size);
    h_surfaceColors = (Vec3f*)malloc(vec_store_size);
    h_inter_spheres = (Sphere**)malloc(sphere_ptr_size);

    // arrays on the GPU
    Vec3f *pixel;
    Sphere *d_spheres;
    Vec3f *d_rays_orig;
    Vec3f *d_rays_dir;
    Sphere **d_inter_spheres;
    Vec3f *d_surfaceColors;

    // allocate GPU memory
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_spheres, size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&pixel, flatArraySize));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_rays_orig, vec_store_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_rays_dir, vec_store_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_surfaceColors, vec_store_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_inter_spheres, sphere_ptr_size));

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_spheres, h_spheres, size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pixel, image, flatArraySize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_rays_orig, h_rays_orig, vec_store_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_rays_dir, h_rays_dir, vec_store_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_surfaceColors, h_surfaceColors, vec_store_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_inter_spheres, h_inter_spheres, sphere_ptr_size, cudaMemcpyHostToDevice));

    //call kernel function and choose dimension of the problem
    dim3 dimGrid(128,72);
    dim3 dimBlock(32, 32);

    //the rendering computation
    #if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
    #endif
    //Calculate the portion of image by layers and combine in the end
    //Automatically synced between each layer
    render_cuda<<<dimGrid, dimBlock>>>(d_spheres, height, width, pixel, d_rays_orig, d_rays_dir, d_surfaceColors, d_inter_spheres);
    //render(spheres);
    
    #if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU calculation time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(image, pixel, flatArraySize, cudaMemcpyDeviceToHost));

    //Free Memory

    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::ofstream ofs("./cuda_untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
               (unsigned char)(std::min(float(1), image[i].y) * 255) <<
               (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
    
    return 0;
}