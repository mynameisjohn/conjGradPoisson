#if ( (defined(__MACH__)) && (defined(__APPLE__)) )   

#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glext.h>
#endif

#include <CL/cl.h>
#include <clAmdBlas.h>
//#include "mkl.h"
#include <omp.h>
#include "../shader/shader.h"
//#include "../solver/conjGrad_MKL.h"
#include "../solver/conjGrad_OCL.h"

#define DIM 1024
#define VERTEX_SHADER "shader/shader.vert"
#define FRAGMENT_SHADER "shader/shader.frag"

Shader shader;

GLuint gvPositionHandle;
GLuint a_TexCoordinate_handle;   //Program handle for the a_TexCoordinate variable
GLuint u_Texture_handle;         //Program handle for the u_Texture variable
GLuint texHandle;                //Program handle for the texture map
int done=0;

void init(void) {
   //initialize shader
   shader.init(VERTEX_SHADER, "shader/shader.frag");

   //Get the position, texture coordinat, and texture sampler handles
   shader.getPosHandle(&gvPositionHandle);
   shader.getTexHandle(&a_TexCoordinate_handle,&u_Texture_handle);
   
   //Generate the texture and assign it to our handle
   glGenTextures(1, &texHandle);
   
   // Bind to the texture in OpenGL
   glBindTexture(GL_TEXTURE_2D, texHandle);
   
   // Set filtering
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   
   // Create a texture of appropriate size with no data (R32F for unclamped floating point texture)
   glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, DIM, DIM, 0, GL_RED, GL_FLOAT, 0x0000);
   shader.bind();

   
}


//The vertices of our screen rectangle
const GLfloat glScreenRectVertices[] =
   {-1.0f,  -1.0f,
    -1.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, -1.0f};

//The texture coordinates for our screen rectangle
const GLfloat glScreenRectTexCoords[] =
   {0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f};

void display (void) {
   //Array used to contain voltage values (passed to conjgrad then used as float texture)
   static float * PXA = (float *)malloc(sizeof(float)*DIM*DIM);
   
   //While conjGrad is still solving, send it PXA
   if (!done) 
     done = solve(PXA,DIM);
   
   //OpenGL nonsense
   glClearColor (0.0,0.0,0.0,1.0);
   glClear (GL_COLOR_BUFFER_BIT);
   glLoadIdentity();  
   
   //Set up our texture
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texHandle);
   glUniform1i(u_Texture_handle, 0);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DIM, DIM, GL_RED, GL_FLOAT, PXA);
   
   //Pass the texture coordinates to the shader
   glVertexAttribPointer(a_TexCoordinate_handle, 2, GL_FLOAT, GL_FALSE, 8, glScreenRectTexCoords);
   glEnableVertexAttribArray(a_TexCoordinate_handle);
   
   //Pass the vertices to the shader
   glVertexAttribPointer(gvPositionHandle, 2, GL_FLOAT, GL_FALSE, 8, glScreenRectVertices);
   glEnableVertexAttribArray(gvPositionHandle);

   //Draw
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
   
   //Swap screen buffer
   glutSwapBuffers();
}

void reshape (int w, int h) {
   glViewport (0, 0, (GLsizei)w, (GLsizei)h);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ();
   gluPerspective (60, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
   glMatrixMode (GL_MODELVIEW);
}

int main (int argc, char **argv) {
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA ); //set up the double buffering
   glutInitWindowSize(DIM, DIM);
   glutInitWindowPosition(100, 100);
   glutCreateWindow("A basic OpenGL Window");
   
   glutDisplayFunc(display);
   glutIdleFunc(display);
   
   glutReshapeFunc(reshape);
   GLenum err = glewInit();
   init();
   
   glutMainLoop();
   
   return 0;
}

