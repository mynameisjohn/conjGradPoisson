attribute vec4 vPosition;
attribute vec2 a_TexCoordinate;
varying vec2 v_TexCoordinate;

void main() {
   v_TexCoordinate = a_TexCoordinate;
   gl_Position = vPosition;
}
