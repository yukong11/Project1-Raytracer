-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Kong Ma
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Implemented Feature
-------------------------------------------------------------------------------
Finished all the basic features:

* Raycasting from a camera into a scene through a pixel grid
* Phong lighting for one point light source
* Diffuse lambertian surfaces
* Raytraced shadows
* Cube intersection testing
* Sphere surface point sampling

Finished following optional features
* Specular reflection 
* Soft shadows and area lights 
* Depth of field
* Supersampled antialiasing   ------use jittering rather than supersampling for efficiency
* Refraction, i.e. glass      ------refraction is not robust
        

-------------------------------------------------------------------------------
Features Analysis
-------------------------------------------------------------------------------
1. Complete Basic ray tracer
   * Implemented basic ray tracer based on local illumination equation to deal with Lambert surface and specular highlight.
   * Todo:  square specular highlight.
2. Soft shadows
   * Soft shadows are achieved by tracing multiple light rays from the intersection point to the light source, rather than one light ray.  To do this, I choose the light ray starts from a random point on area light and ends at the intersection point.
   * However, due to the limitation of the launch time of Kernel function on GPU, in this project, I can only achieve at most 25 random rays in one illumination test when enable reflection rays. Therefore, it takes more iterations for the image to converge. 
3. Anti-aliasing
   * To deal with the aliasing, we can use super-sampling and jittering method. In this project I am only using the jittering method because the result image we got after one iteration is the average result of previous iterations. With jittering we can achieve a simple effects of super-sampling.
   * Todo: Although jittering can give us relative good result, however if there is a refraction object in the scene, the direction of light is deviate to a large distance, the aliasing problem is exaggerated. Jittering cannot solve the problem;
4. Depth of field
   * In this project, I am using the  method introduced by Cook et al. in 1984 in  Distributed Ray Tracing. The result is generally good, but it takes more iteration and longer time to converge.
   * Maybe a post processing method can be used to achieve real-time result. 
5. Reflection and Refraction
   * The calculations of reflectance and  transmittance are based on the Fresnel equation with the assumption that  light is unpolarised (containing an equal mix of s- and p-polarisations).
   * However, a big problem is that,  I didn't take the deviation effect of refraction surface into consideration when sampling whether a point can be illuminated by the light source. I think a different approach could be used to do this illumination test, which could solve the refraction deviation effects and square specular highlight at same time. 

-------------------------------------------------------------------------------
BLOG LINK:
-------------------------------------------------------------------------------
http://gpuprojects.blogspot.com/

-------------------------------------------------------------------------------
Instruction on buiding and running
-------------------------------------------------------------------------------
Default Setting is softshadow result without depth of field effect.

1. if you machine gives warnings such as "Kernel failed! unknown error!" or "Kernel failed! the launch timed out and was terminated". It is possible that the soft shadow takes too long to compute in your machine, you can decrease the value in "__constant__ int rayNumbers=10;".

2. if you don't want the soft shadow effects,you can make the value in  "__constant__ int rayNumbers=10;" to 1, and change the value "__constant__ bool softShadow=true;" to false

3. if you want to see depth of field result, uncomment the line "//#define DEPTHOFFIELD"
