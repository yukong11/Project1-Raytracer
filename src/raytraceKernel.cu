// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>


//#define DEPTHOFFIELD
float Lensdiameter=200;
__constant__ int rayNumbers=10;
__constant__ bool softShadow=true;

//bool firstRayCached=false;
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(-1,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}


__global__ void calculateRaycastFromCameraKernel(cameraData cam, float time,ray* rayArray){

	thrust::default_random_engine rng(hash(time));
	thrust::uniform_real_distribution<float> u01(-1,1);
	float noisex =((float)u01(rng))*0.5f;
	float noisey =((float)u01(rng))*0.5f;

	ray r;
	r.origin = cam.position;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	if((x<=cam.resolution.x )&&( y<=cam.resolution.y)){
    float y1=cam.resolution.y-y;
	float x1=cam.resolution.x-x;
	glm::vec3 A = glm::cross(cam.view,cam.up); //A= view^up
	float ALength=glm::length(A);
	glm::vec3 B =  glm::cross(A,cam.view);	//B <- A * C
	float BLength=glm::length(B);
    glm::vec3 M = cam.position + cam.view;	//M=E+C
	float viewLength=glm::length(cam.view);
	glm::vec3 H = A*viewLength * (float)tan(cam.fov.x*(PI/180.0f))/ ALength; //H <- (A|C|tan)/|A|
	glm::vec3 V = B*viewLength *(float)tan(cam.fov.y*(PI/180.0f)) / BLength;   // V <- (B|C|tan)/|B|  
	//glm::vec3 P=M+(2*((float)x1/(float)(cam.resolution.x-1))-1)*H+(2*(float)y1/(float)(cam.resolution.y-1)-1)*V;
	glm::vec3 P=M+(2*((float)(x1+noisex)/(float)(cam.resolution.x-1))-1)*H+(2*(float)(y1+noisey)/(float)(cam.resolution.y-1)-1)*V;
	glm::vec3 D=P-cam.position;
	r.direction=glm::normalize(D);
    rayArray[index]=r;
	}
	return;
}


//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


__global__ void raytracefromCameraKernel(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int  numberOfGeoms,material* materials,ray* cudaFirstRays,rayData* rayList){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x>resolution.x )||( y>cam.resolution.y))return;
  ray r=cudaFirstRays[index];
 
  glm::vec3 finalColor=glm::vec3(0.0f,0.0f,0.0f);
    
	//find first intersection
	float distance=-1.0f;
	glm::vec3 interestPoint=glm::vec3(0,0,0);
	glm::vec3 normal=glm::vec3(0,0,0);
	int geoID=-1;
	for(int i=0; i<numberOfGeoms; i++){
		float tempdistance=-1.0;
		glm::vec3 tempInterestPoint=glm::vec3(0,0,0);
	    glm::vec3 tempNormal=glm::vec3(0,0,0);
		if(geoms[i].type==SPHERE){
			tempdistance=sphereIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}else if(geoms[i].type==CUBE){
			tempdistance=boxIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}

		if((abs(distance+1.0f)<1e-3)&&(tempdistance>0.001f)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}else if((tempdistance>0.001f)&&(tempdistance<distance)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}

		}

	//can not find intersection ,ray ends
	if(geoID==-1){
		if((x<=resolution.x) &&( y<=resolution.y)){
		 rayList[index].dirty=false;
		 rayList[index].softshadow=false;
	 }	
		return;
	}

	material m=materials[geoms[geoID].materialid];
	if(m.emittance>0){///light source
	     if((x<=resolution.x) && (y<=resolution.y)){
		 colors[index] =m.color;
		 rayList[index].dirty=false;
		  rayList[index].softshadow=false;
	 }
		 return;
	}

	//calculateReflectionDirection(normal,r.direction)
	rayData newRayData;
	newRayData.newray.direction=r.direction;  //p1-2*(p1*pn)*pn
	newRayData.newray.origin=interestPoint;
	newRayData.reflectionCoeff=0.5f;
	newRayData.normal=normal;
	newRayData.softshadow=true;
	newRayData.dirty=false;
	newRayData.ID=geoID;
	rayList[index]=newRayData;
	////local=ambience+diffuse+specular
	//glm::vec3 tempColor=glm::vec3(0.0f,0.0f,0.0f);//ambience

 // 	for(int i=0; i<numberOfGeoms; i++){  
	//	if((materials[geoms[i].materialid].emittance>0)){ // i points to light position
	//		tempColor=glm::vec3(0.0f,0.0f,0.0f);//ambience
	//		glm::vec3 lightColor=materials[geoms[i].materialid].color;
	//		
	//		ray lightray;
	//		lightray.origin=interestPoint;
	//		lightray.direction=glm::normalize(geoms[i].translation-interestPoint);
	//		//int rayNumbers=3;
	//		//int lightRayCount=3;
		//	//while(lightRayCount>0){
		//	//	tempColor=glm::vec3(0.0f,0.0f,0.0f);
		//	//	glm::vec3 lightpos;
		//	//	if(geoms[i].type==SPHERE){
		//	//		//lightpos=getRandomPointOnSphere(geoms[i],time);
		//	//		}else if(geoms[i].type==CUBE){
		//	//		lightpos=getRandomPointOnCube(geoms[i],time);
		//	//		}
		//	//     lightray.direction=glm::normalize(lightpos-interestPoint);

		//		if(glm::dot(lightray.direction,normal)>=0){
		//	
		//		float interdistance=-1.0f;
		//		glm::vec3 a,b;
		//		for(int j=0; j<numberOfGeoms; j++){ //intersect with object j
		//			float tempt=0;
		//			if((j!=i)){
		//			if(geoms[j].type==SPHERE){
		//			tempt=sphereIntersectionTest(geoms[j],lightray,a,b);
		//			}else if(geoms[j].type==CUBE){
		//			tempt=boxIntersectionTest(geoms[j],lightray,a,b);}
		//			}
		//       
		//			if((abs(interdistance+1.0f)<1e-3)&&(tempt>0.001f)){
		//				interdistance=tempt;
		//			}else if((tempt>0.001f)&&(tempt<interdistance)){
		//				interdistance=tempt;
		//			}
		//		}

		//		if((abs(interdistance+1.0f)<1e-3)||(interdistance+0.2f>glm::length(geoms[i].translation-interestPoint)))   //if view by light
		//		{
		//			//diffuse
		//			float diffuseTerm=0.0f;
		//			if(glm::dot(normal,lightray.direction)>0.0f)
		//				diffuseTerm=glm::dot(normal,lightray.direction);
		//			tempColor+=diffuseTerm*glm::vec3(lightColor.x*m.color.x,lightColor.y*m.color.y,lightColor.z*m.color.z)/*/float(rayNumbers)*/;
		//	
		//			//specular
		//			if(m.specularExponent>0.001){
		//			glm::vec3 viewDirection=glm::normalize(cam.position-interestPoint);
		//			glm::vec3 lightReflection=calculateReflectionDirection(normal,-lightray.direction);
		//			float specularTerm=0.0f;
		//			if(glm::dot(viewDirection,lightReflection)>0.0f)
		//			specularTerm=pow(glm::dot(viewDirection,lightReflection),m.specularExponent);
		//			tempColor+=specularTerm*glm::vec3(lightColor.x*m.specularColor.x,lightColor.y*m.specularColor.y,lightColor.z*m.specularColor.z)/*/float(rayNumbers)*/;	
		//			}
		//			
		//		 }else if(interdistance+0.2f<glm::length(geoms[i].translation-interestPoint)){

		//			 //rayList[index].newray.origin=interestPoint;
		//			 //rayList[index].normal=normal;
		//			 //rayList[index].softshadow=true;
		//			 //return;
		//		}

		//		}
		//finalColor+=tempColor;
		//	//--lightRayCount;
		//  // }//end of one light ray
		// }//end of one light object
	//}//end of light object list
	//
	// if((x<=resolution.x) && (y<=resolution.y)){
	//  colors[index] =finalColor;
	//}

 // 	if(m.hasReflective){
 //   rayData newRayData;
	//newRayData.newray.direction=calculateReflectionDirection(normal,r.direction);  //p1-2*(p1*pn)*pn
	//newRayData.newray.origin=interestPoint;
	//newRayData.reflectionCoeff=0.5f;
	//newRayData.dirty=true;
	//newRayData.x=x;
	//newRayData.y=y;
	////newRayData.normal=normal;
	////newRayData.softshadow=false;
	//rayList[index]=newRayData;
	//}

}


__global__ void iterationRaytrace(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
	staticGeom* geoms, int numberOfGeoms,material* materials,rayData* rayList){
 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x>resolution.x )&&(y>resolution.y))return;
  rayData rd=rayList[index];
  if(!rd.dirty)return;

  ray r=rd.newray;

  glm::vec3 finalColor=glm::vec3(0.0f,0.0f,0.0f);
    
	//find first intersection
	float distance=-1.0f;
	glm::vec3 interestPoint=glm::vec3(0,0,0);
	glm::vec3 normal=glm::vec3(0,0,0);
	int geoID=-1;
	for(int i=0; i<numberOfGeoms; i++){
		float tempdistance=-1.0;
		glm::vec3 tempInterestPoint=glm::vec3(0,0,0);
	    glm::vec3 tempNormal=glm::vec3(0,0,0);
		if(geoms[i].type==SPHERE){
			tempdistance=sphereIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}else if(geoms[i].type==CUBE){
			tempdistance=boxIntersectionTest(geoms[i],r,tempInterestPoint,tempNormal);
		}

		if((abs(distance+1.0f)<1e-3)&&(tempdistance>0.001f)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}else if((tempdistance>0.001f)&&(tempdistance<distance)){
			distance=tempdistance;
			normal=tempNormal;
			interestPoint=tempInterestPoint;
			geoID=i;
		}

		}

	//can not find intersection ,ray ends
	if(geoID==-1){
		if((x<=resolution.x )&&(y<=resolution.y)){
			rayList[index].dirty=false;
	 }	
		return;
	}

	material m=materials[geoms[geoID].materialid];
	if(m.emittance>0){///light source
	 if((x<=resolution.x) && (y<=resolution.y)){
		 colors[index]=colors[index]+m.color*rd.reflectionCoeff;
		 rayList[index].dirty=false;
	 }
		 return;
	}


	//local=ambience+diffuse+specular
	glm::vec3 tempColor=glm::vec3(0.0f,0.0f,0.0f);//ambience
  	for(int i=0; i<numberOfGeoms; i++){  
		if((materials[geoms[i].materialid].emittance>0)){ // i points to light position
			tempColor=glm::vec3(0.0f,0.0f,0.0f);
			glm::vec3 lightColor=materials[geoms[i].materialid].color;
			
			ray lightray;
			lightray.origin=interestPoint;
			lightray.direction=glm::normalize(geoms[i].translation-interestPoint);
			
			int lightRayCount=rayNumbers;
			while(lightRayCount>0){
				tempColor=glm::vec3(0.0f,0.0f,0.0f);
				glm::vec3 lightpos;
				if(geoms[i].type==SPHERE){
					lightpos=getRandomPointOnSphere(geoms[i],time);
					}else if(geoms[i].type==CUBE){
					lightpos=getRandomPointOnCube(geoms[i],time);
					}
				if(softShadow)
			     lightray.direction=glm::normalize(lightpos-interestPoint);
			   //if(glm::dot(lightray.direction,normal)<0) break;
			
				float interdistance=-1.0f;
				glm::vec3 a,b;
				for(int j=0; j<numberOfGeoms; j++){ //intersect with object j
					float tempt=0;
					if((j!=i)){
					if(geoms[j].type==SPHERE){
					tempt=sphereIntersectionTest(geoms[j],lightray,a,b);
					}else if(geoms[j].type==CUBE){
					tempt=boxIntersectionTest(geoms[j],lightray,a,b);}
					}
		       
					if((abs(interdistance+1.0f)<1e-3)&&(tempt>0.001f)){
						interdistance=tempt;
					}else if((tempt>0.001f)&&(tempt<interdistance)){
						interdistance=tempt;
					}
				}

				if((abs(interdistance+1.0f)<1e-3)||(interdistance+0.2f>glm::length(geoms[i].translation-interestPoint)))   //if view by light
				{
					//diffuse
					float diffuseTerm=0.0f;
					if(glm::dot(normal,lightray.direction)>0.0f)
						diffuseTerm=glm::dot(normal,lightray.direction);
					tempColor+=diffuseTerm*glm::vec3(lightColor.x*m.color.x,lightColor.y*m.color.y,lightColor.z*m.color.z)/float(rayNumbers);
			
					//specular
					if(m.specularExponent>0.001){
					glm::vec3 viewDirection=glm::normalize(cam.position-interestPoint);
					glm::vec3 lightReflection=calculateReflectionDirection(normal,-lightray.direction);
					float specularTerm=0.0f;
					if(glm::dot(viewDirection,lightReflection)>0.0f)
					specularTerm=pow(glm::dot(viewDirection,lightReflection),m.specularExponent);
					//tempColor+=specularTerm*glm::vec3(lightColor.x*m.specularColor.x,lightColor.y*m.specularColor.y,lightColor.z*m.specularColor.z)/float(rayNumbers);	
					tempColor+=specularTerm*lightColor/float(rayNumbers);	
					}
					
					
				 }
				//}
			finalColor+=tempColor;
			--lightRayCount;
		   }//end of one light ray
		 }//end of one light object
		}//end of light object list


  if((x<=resolution.x)&&(y<=resolution.y)){
	  colors[index]+=finalColor*rd.reflectionCoeff;
	
   }

  if((!m.hasReflective)&&(!m.hasRefractive)){
	  return;
  }

  glm::vec3 reflectedDirection,transmittedDirection;
  Fresnel fr;
	if(glm::dot(normal,rd.newray.direction)<0.001)
		fr=calculateFresnel(normal,rd.newray.direction,1.0f,m.indexOfRefraction,reflectedDirection,transmittedDirection);	
	else 
		fr=calculateFresnel(-normal,rd.newray.direction,m.indexOfRefraction,1.0f,reflectedDirection,transmittedDirection);	

  bool reflectiveRay=false,refractiveRay=false;
  if((m.hasReflective)&&(m.hasRefractive)){
	
	  if(abs(fr.transmissionCoefficient)<1e-3){
		  reflectiveRay=true;
	  }else{
		thrust::default_random_engine rng(hash(time));
		thrust::uniform_real_distribution<float> u01(0,1);
		 float russianRoulette = (float)u01(rng);
		if(russianRoulette<0.5)
		 reflectiveRay=true;
		else
		 refractiveRay=true;
	  }
	 }else if(m.hasReflective){
		 reflectiveRay=true;
	 }else if(m.hasRefractive){
		 refractiveRay=true;
	 }else
		 return;
	
  if(reflectiveRay){
   rayData newRayData;
	newRayData.newray.direction=reflectedDirection;  //p1-2*(p1*pn)*pn
	newRayData.newray.origin=interestPoint;
	newRayData.reflectionCoeff=rd.reflectionCoeff*fr.reflectionCoefficient;
	newRayData.dirty=true;
	newRayData.x=x;
	newRayData.y=y;
	rayList[index]=newRayData;
	return;
	}
	
	if(refractiveRay){
    rayData newRayData;
	newRayData.newray.direction=transmittedDirection;  //p1-2*(p1*pn)*pn
	newRayData.newray.origin=interestPoint;
	newRayData.reflectionCoeff=rd.reflectionCoeff*fr.transmissionCoefficient;
	newRayData.dirty=true;
	newRayData.x=x;
	newRayData.y=y;
	rayList[index]=newRayData;
	return;
	}

}


 __global__ void mergeImage(glm::vec2 resolution,glm::vec3* previousColors,glm::vec3* currentColors,float time){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
   if((x<=resolution.x) && (y<=resolution.y)){
	    glm::vec3 currentColor=currentColors[index];
		 glm::vec3 previousColor=previousColors[index];

	   currentColors[index]=currentColor/time+previousColor*(time-1.0f)/time;
   }

 }

  
__global__ void  calculateSoftShadow(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, 
                            staticGeom* geoms, int  numberOfGeoms,material* materials,ray* cudaFirstRays,rayData* rayList){ 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x>resolution.x )||( y>cam.resolution.y))return;
   rayData rd=rayList[index];
   if(rd.softshadow==false)return;
   glm::vec3 normal=rd.normal;
   glm::vec3 interestPoint=rd.newray.origin;
   material m=materials[geoms[rd.ID].materialid];
   glm::vec3 finalColor=glm::vec3(0.0f,0.0f,0.0f);

//local=ambience+diffuse+specular
	glm::vec3 tempColor=glm::vec3(0.0f,0.0f,0.0f);//ambience
  	for(int i=0; i<numberOfGeoms; i++){  
		if((materials[geoms[i].materialid].emittance>0)){ // i points to light position
			 tempColor=glm::vec3(0.0f,0.0f,0.0f);
			glm::vec3 lightColor=materials[geoms[i].materialid].color;
			
			ray lightray;
			lightray.origin=interestPoint;
			lightray.direction=glm::normalize(geoms[i].translation-interestPoint);
			int lightRayCount=rayNumbers;
			while(lightRayCount>0){
				tempColor=glm::vec3(0.0f,0.0f,0.0f);
				glm::vec3 lightpos;
				if(geoms[i].type==SPHERE){
					lightpos=getRandomPointOnSphere(geoms[i],time);
					}else if(geoms[i].type==CUBE){
					lightpos=getRandomPointOnCube(geoms[i],time);
					}
				if(softShadow)
			     lightray.direction=glm::normalize(lightpos-interestPoint);

				//if(glm::dot(lightray.direction,normal)>=0){
			
				float interdistance=-1.0f;
				glm::vec3 a,b;
				for(int j=0; j<numberOfGeoms; j++){ //intersect with object j
					float tempt=0;
					if((j!=i)){
					if(geoms[j].type==SPHERE){
					tempt=sphereIntersectionTest(geoms[j],lightray,a,b);
					}else if(geoms[j].type==CUBE){
					tempt=boxIntersectionTest(geoms[j],lightray,a,b);}
					}
		       
					if((abs(interdistance+1.0f)<1e-3)&&(tempt>0.001f)){
						interdistance=tempt;
					}else if((tempt>0.001f)&&(tempt<interdistance)){
						interdistance=tempt;
					}
				}

				if((abs(interdistance+1.0f)<1e-3)||(interdistance+0.2f>glm::length(geoms[i].translation-interestPoint)))   //if view by light
				{
					//diffuse
					float diffuseTerm=0.0f;
					if(glm::dot(normal,lightray.direction)>0.0f)
						diffuseTerm=glm::dot(normal,lightray.direction);
					tempColor+=glm::dot(normal,lightray.direction)*glm::vec3(lightColor.x*m.color.x,lightColor.y*m.color.y,lightColor.z*m.color.z)/float(rayNumbers);
			
					//specular
					if(m.specularExponent>0.001){
					glm::vec3 viewDirection=glm::normalize(cam.position-interestPoint);
					glm::vec3 lightReflection=calculateReflectionDirection(normal,-lightray.direction);
					float specularTerm=0.0f;
					if(glm::dot(viewDirection,lightReflection)>0.0f)
					specularTerm=pow(glm::dot(viewDirection,lightReflection),m.specularExponent);
					//tempColor+=specularTerm*glm::vec3(lightColor.x*m.specularColor.x,lightColor.y*m.specularColor.y,lightColor.z*m.specularColor.z)/float(rayNumbers);	
					tempColor+=specularTerm*lightColor/float(rayNumbers);	
					}
					
					
				 }
				//}
			finalColor+=tempColor;
			--lightRayCount;
		   }//end of one light ray
		 }//end of one light object
		}//end of light object list
	
	 if((x<=resolution.x) && (y<=resolution.y)){
	  colors[index] =finalColor;
	}

  if((!m.hasReflective)&&(!m.hasRefractive)){
	  return;
  }

  glm::vec3 reflectedDirection,transmittedDirection;
  Fresnel fr;
	  if(glm::dot(normal,rd.newray.direction)<0.001)
		fr=calculateFresnel(normal,rd.newray.direction,1.0f,m.indexOfRefraction,reflectedDirection,transmittedDirection);	
	else 
		fr=calculateFresnel(-normal,rd.newray.direction,m.indexOfRefraction,1.0f,reflectedDirection,transmittedDirection);	

  bool reflectiveRay=false,refractiveRay=false;
  if((m.hasReflective)&&(m.hasRefractive)){
	  if(abs(fr.transmissionCoefficient)<0.1){
		  reflectiveRay=true;
	  }else{
		thrust::default_random_engine rng(hash(time));
		thrust::uniform_real_distribution<float> u01(0,1);
		 float russianRoulette = (float)u01(rng);
		if(russianRoulette<0.5)
		 reflectiveRay=true;
		else
		 refractiveRay=true;
	  }
	 }else if(m.hasReflective){
		 reflectiveRay=true;
	 }else if(m.hasRefractive){
		 refractiveRay=true;
	 }
	
  	if(reflectiveRay){
    rayData newRayData;
	newRayData.newray.direction=reflectedDirection;  //p1-2*(p1*pn)*pn
	newRayData.newray.origin=interestPoint;
	newRayData.reflectionCoeff=fr.reflectionCoefficient;
	newRayData.dirty=true;
	newRayData.x=x;
	newRayData.y=y;
	rayList[index]=newRayData;
	return;
	}
	
	if(refractiveRay){
    rayData newRayData;
	newRayData.newray.direction=transmittedDirection;  //p1-2*(p1*pn)*pn
	newRayData.newray.origin=interestPoint;
	newRayData.reflectionCoeff=fr.transmissionCoefficient;
	newRayData.dirty=true;
	newRayData.x=x;
	newRayData.y=y;
	rayList[index]=newRayData;
	return;
	}


}
//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, ray* firstRays){
  
  int traceDepth =10; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 25;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
 // cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
   //--------------------------------
  //package materials
  material* materialList=new material[numberOfMaterials];

  for(int i=0;i<numberOfMaterials;i++){
	  material newmaterial;
	  newmaterial.color=materials[i].color;
	  newmaterial.specularExponent=materials[i].specularExponent;
	  newmaterial.specularColor=materials[i].specularColor;
	  newmaterial.hasReflective=materials[i].hasReflective;
	  newmaterial.hasRefractive=materials[i].hasRefractive;
	  newmaterial.indexOfRefraction=materials[i].indexOfRefraction;
	  newmaterial.hasScatter=materials[i].hasScatter;
	  newmaterial.absorptionCoefficient=materials[i].absorptionCoefficient;
	  newmaterial.reducedScatterCoefficient=materials[i].reducedScatterCoefficient;
	  newmaterial.emittance=materials[i].emittance;

	  materialList[i]=newmaterial;
  }

  material* cudamatrials=NULL;
  cudaMalloc((void**)&cudamatrials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamatrials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
	 	  
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

    //first Rays cudaMemory Pointer
  ray* cudaFirstRays = NULL;
  cudaMalloc((void**)&cudaFirstRays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));

  // previous image cudaMemory Pointer
  glm::vec3* previousImage = NULL;
  cudaMalloc((void**)&previousImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

  //saved first ray Color cudaMemory Pointer
  rayData* cudaRayList = NULL;
  cudaMalloc((void**)&cudaRayList, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(rayData));

  #ifdef DEPTHOFFIELD
	thrust::default_random_engine rng(hash((float)iterations));
    thrust::uniform_real_distribution<float> u01(-1,1);
	float xDist=(float)u01(rng);
	float yDist=(float)u01(rng);

	float length=abs(glm::dot(glm::vec3(geomList[5].translation-cam.position),cam.view));
	glm::vec3 focalPos=cam.position+cam.view*length;
	cam.position+=glm::vec3(xDist*Lensdiameter*1/cam.resolution.x,yDist*Lensdiameter*1/cam.resolution.y,0.0f);
	cam.view=glm::normalize(focalPos-cam.position);
	int NumberOfCamPos=10;
	int CamPosCount=NumberOfCamPos;
	while(CamPosCount>0){
  #endif

// save the first Ray Direction
 
	// if(!firstRayCached){
	calculateRaycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(cam,(float)iterations,cudaFirstRays);
	cudaMemcpy(firstRays, cudaFirstRays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray), cudaMemcpyDeviceToHost);
	// firstRayCached=true;
//  }else{  
//  cudaMemcpy( cudaFirstRays, firstRays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray), cudaMemcpyHostToDevice);
//  }

   //calculate first ray color
  raytracefromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamatrials,cudaFirstRays,cudaRayList);
  checkCUDAError("Kernel failed! 2");
  calculateSoftShadow<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamatrials,cudaFirstRays,cudaRayList);
  checkCUDAError("Kernel failed! 3");

  //iteration kernel launches
  int currDepth=1;
  while((currDepth<=traceDepth)){ 
	iterationRaytrace<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamatrials,cudaRayList);
	checkCUDAError("Kernel failed! 4");
	currDepth++;
	
  }
 
  checkCUDAError("Kernel failed! 5");

  //combine several iteration together
  cudaMemcpy( previousImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  checkCUDAError("Kernel failed! 6");

  mergeImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,previousImage,cudaimage,(float)iterations),
  checkCUDAError("Kernel failed! 7");
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  #ifdef DEPTHOFFIELD
	CamPosCount--;
		}
  #endif

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree(cudaFirstRays);
  cudaFree(cudaRayList);
  cudaFree( cudageoms );
  cudaFree(previousImage);

  delete geomList;
  delete materialList;

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
