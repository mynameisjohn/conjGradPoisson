#define M_PI 3.1415926535897932384626433832795

precision mediump float;
varying vec2 v_TexCoordinate;
uniform sampler2D u_Texture;

void main(){
   vec4 voltageColor = texture2D(u_Texture,v_TexCoordinate);
   float v = test.x; //red component of the color we passed in
   float t,r,g,b;
   t=(5.0f*(M_PI/2.0f-atan(v)))/3.0f;
   r=0.5f*(2.0f*cos(t)+1.0f);
   g=0.5f*(2.0f*cos(t-(2.0f)*M_PI/(3.0f))+1.0f);
   b=0.5f*(2.0f*cos(t+(2.0f)*M_PI/3.0f)+1.0f);

   gl_FragColor = vec4(r,g,b,1.0f); //vec4 (0.5f*sin(v)+0.5f,0.0f,0.0f, 1.0);
}
