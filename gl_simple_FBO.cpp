/*-----------------------------------------------------------------------
    Copyright (c) 2013, NVIDIA. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Neither the name of its contributors may be used to endorse 
       or promote products derived from this software without specific
       prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    feedback to tlorach@nvidia.com (Tristan Lorach)
*/ //--------------------------------------------------------------------
#pragma GCC diagnostic warning "-fpermissive"
#include "main.h"
#include "nv_helpers_gl/WindowInertiaCamera.h"
#include "nv_helpers_gl/GLSLProgram.h"

#include "bk3dEx.h" // a baked binary format for few models

#include "SvCMFCUI.h"

#include "nv_math/nv_math_glsltypes.h"
//-----------------------------------------------------------------------------
// Derive the Window for this sample
//-----------------------------------------------------------------------------
class MyWindow: public WindowInertiaCamera
{
	bool		m_validated;
public:
	MyWindow() : m_validated(false) {}
    virtual bool init();
    virtual void shutdown();
    virtual void reshape(int w, int h);
    //virtual void motion(int x, int y);
    //virtual void mousewheel(short delta);
    //virtual void mouse(NVPWindow::MouseButton button, ButtonAction action, int mods, int x, int y);
    //virtual void menu(int m);
    virtual void keyboard(MyWindow::KeyCode key, ButtonAction action, int mods, int x, int y);
    virtual void keyboardchar(unsigned char key, int mods, int x, int y);
    //virtual void idle();
    virtual void display();

	void renderScene();
};

/////////////////////////////////////////////////////////////////////////
// grid Floor
static const char *g_glslv_grid = 
"#version 330\n"
"uniform mat4 mWVP;\n"
"layout(location=0) in  vec3 P;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"void main() {\n"
"   gl_Position = mWVP * vec4(P, 1.0);\n"
"}\n"
;
static const char *g_glslf_grid = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform vec3 diffuse;"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   outColor = vec4(diffuse,1);\n"
"}\n"
;

/////////////////////////////////////////////////////////////////////////
// Mesh
static const char *g_glslv_mesh = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform mat4 mWVP;\n"
"uniform mat4 mVP;\n"
"layout(location=0) in  vec3 P;\n"
"layout(location=1) in  vec3 N;\n"
"layout(location=1) out vec3 outN;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"void main() {\n"
"   outN = N;\n"
"   gl_Position = mWVP * vec4(P, 1.0);\n"
"}\n"
;
static const char *g_glslf_mesh = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform vec3 diffuse;"
"uniform vec3 lightDir;"
"layout(location=1) in  vec3 N;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"\n"
"   float d1 = max(0.0, dot(N, lightDir) );\n"
"   float d2 = 0.6 * max(0.0, dot(N, -lightDir) );\n"
"   outColor = vec4(diffuse * (d2 + d1),1);\n"
"}\n"
;

/////////////////////////////////////////////////////////////////////////
// FBO resolve

static const char *g_glslv_Tc = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform ivec2 viewportSz;\n"
"layout(location=0) in  ivec2 P;\n"
"layout(location=0) out vec2 TcOut;\n"
"out gl_PerVertex {\n"
"    vec4  gl_Position;\n"
"};\n"
"void main() {\n"
"   TcOut = vec2(P);\n"
"   gl_Position = vec4(vec2(P)/vec2(viewportSz)*2.0 - 1.0, 0.0, 1.0);\n"
"}\n"
;
// for sampling MSAA Texture
static const char *g_glslf_texMSAA = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform sampler2DMS samplerMS;\n"
"layout(location=0) in vec2 Tc;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   vec4 c = vec4(0);\n"
"   for(int i=0; i<8; i++)\n" // assuming MSAA 8x
"       c += texelFetch(samplerMS, ivec2(Tc), i);\n"
"   outColor = c / 8.0;\n"
"}\n"
;

static const char *g_glslf_tex = 
"#version 330\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform sampler2D s;\n"
"layout(location=0) in vec2 Tc;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   outColor = texelFetch(s, ivec2(Tc), 0);\n"
"}\n"
;

// for sampling MSAA Texture
static const char *g_glslf_ImageMSAA = 
"#version 420\n"
//"#extension GL_ARB_shader_image_load_store : enable\n"
//"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform layout(rgba8) image2DMS imageMS;\n"
"layout(location=0) in vec2 Tc;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   vec4 c = vec4(0);\n"
"   for(int i=0; i<8; i++)\n" // assuming MSAA 8x
"       c += imageLoad(imageMS, ivec2(Tc), i);\n"
"   outColor = c / 8.0;\n"
"}\n"
;

static const char *g_glslf_Image = 
"#version 420\n"
//"#extension GL_ARB_shader_image_load_store : enable\n"
//"#extension GL_ARB_separate_shader_objects : enable\n"
"uniform layout(rgba8) image2D image;\n"
"layout(location=0) in vec2 Tc;\n"
"layout(location=0) out vec4 outColor;\n"
"void main() {\n"
"   outColor = imageLoad(image, ivec2(Tc));\n"
"}\n"
;


GLSLProgram g_progGrid;
GLSLProgram g_progMesh;

GLSLProgram g_progCopyTexMSAA;
GLSLProgram g_progCopyTex;
GLSLProgram g_progCopyImageMSAA;
GLSLProgram g_progCopyImage;

GLuint      g_vboGrid = 0;
GLuint      g_vboQuad = 0;

GLuint      g_vao = 0;

// FBO Stuff
GLuint fboSz[2] = {0,0};
GLuint textureRGBA, textureRGBAMS;
GLuint rbRGBA, rbRGBAMS;
GLuint rbDST, rbDSTMS;
GLuint fboTexMS, fboTex, fboRbMS, fboRb;
enum FboMode {
    RENDERTOTEXMS = 0,
    RENDERTOTEX,
    RENDERTORBMS,
    RENDERTORB,
};
FboMode fboMode;

enum BlitMode {
    RESOLVEWITHBLIT = 0,
    RESOLVEWITHSHADERTEX,
    RESOLVEWITHSHADERIMAGE,
};
BlitMode blitMode;

//
// Camera animation: captured using '1' in the sample. Then copy and paste...
//
struct CameraAnim {    vec3f eye, focus; };
static CameraAnim s_cameraAnim[] = {
{vec3f(0.03, 0.58, -1.53), vec3f(0.05, 0.06, 0.02)},
{vec3f(-1.38, 0.39, -0.68), vec3f(0.05, 0.06, 0.02)},
{vec3f(-1.33, 0.56, 0.73), vec3f(0.05, 0.06, 0.02)},
{vec3f(0.10, 0.76, 1.50), vec3f(0.05, 0.06, 0.02)},
{vec3f(2.49, 1.51, -0.19), vec3f(0.54, 0.13, 0.26)},
{vec3f(1.26, 1.82, -2.57), vec3f(0.17, 0.19, -1.16)},
{vec3f(0.60, 1.12, -3.34), vec3f(0.17, 0.19, -1.16)},
{vec3f(-0.02, 0.11, -0.69), vec3f(0.04, -0.04, 0.17)},
{vec3f(-0.05, 0.27, -1.66), vec3f(0.04, -0.04, 0.17)}, // 9 items
};
static int     s_cameraAnimItem     = 0;
static int     s_cameraAnimItems    = 9;
#define ANIMINTERVALL 1.5f
static float   s_cameraAnimIntervals= ANIMINTERVALL;
static bool    s_bCameraAnim        = true;


//---------------------- 3D Model ---------------------------------------------
#ifdef NOGZLIB
#   define MODELNAME "NV_Shaderball_v134.bk3d"
#else
#   define MODELNAME "NV_Shaderball_v134.bk3d.gz"
#endif
bk3d::FileHeader * meshFile;
vec3f g_posOffset = vec3f(0,0,0);
float g_scale = 1.0f;

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void sample_print(int level, const char * txt)
{
#ifdef USESVCUI
    logMFCUI(level, txt);
#else
#endif
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// FBO stuff
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

bool CheckFramebufferStatus()
{
	GLenum status;
	status = (GLenum) glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch(status) {
		case GL_FRAMEBUFFER_COMPLETE:
			return true;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			LOGE("Unsupported framebuffer format\n");
			assert(!"Unsupported framebuffer format");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			LOGE("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n");
			assert(!"GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			LOGE("Framebuffer incomplete, missing attachment\n");
			assert(!"Framebuffer incomplete, missing attachment");
			break;
		//case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
		//	PRINTF(("Framebuffer incomplete, attached images must have same dimensions\n"));
		//	assert(!"Framebuffer incomplete, attached images must have same dimensions");
		//	break;
		//case GL_FRAMEBUFFER_INCOMPLETE_FORMATS:
		//	PRINTF(("Framebuffer incomplete, attached images must have same format\n"));
		//	assert(!"Framebuffer incomplete, attached images must have same format");
		//	break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			LOGE("Framebuffer incomplete, missing draw buffer\n");
			assert(!"Framebuffer incomplete, missing draw buffer");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			LOGE("Framebuffer incomplete, missing read buffer\n");
			assert(!"Framebuffer incomplete, missing read buffer");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			LOGE("GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE\n");
			assert(!"GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE");
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			LOGE("GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS\n");
			assert(!"GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");
			break;
		default:
			LOGE("Error %x\n", status);
			assert(!"unknown FBO Error");
			break;
	}
    return false;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createTexture(int w, int h, int samples, int coverageSamples, GLenum intfmt, GLenum fmt)
{
    GLuint		textureID;
	glGenTextures(1, &textureID);
    if(samples <= 1)
    {
	    glBindTexture( GL_TEXTURE_2D, textureID);
	    glTexImage2D( GL_TEXTURE_2D, 0, intfmt, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
	    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
	    glBindTexture( GL_TEXTURE_2D_MULTISAMPLE, textureID);
        if(coverageSamples > 1)
        {
            glTexImage2DMultisampleCoverageNV(GL_TEXTURE_2D_MULTISAMPLE, coverageSamples, samples, intfmt, w, h, GL_TRUE);
        } else {
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, intfmt, w,h, GL_TRUE);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    return textureID;
}
//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createTextureRGBA8(int w, int h, int samples, int coverageSamples)
{
    return createTexture(w, h, samples, coverageSamples, GL_RGBA8, GL_RGBA);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createRenderBuffer(int w, int h, int samples, int coverageSamples, GLenum fmt)
{
    int query;
    GLuint rb;
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	if (coverageSamples) 
	{
		glRenderbufferStorageMultisampleCoverageNV( GL_RENDERBUFFER, coverageSamples, samples, fmt,
													w, h);
		glGetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_COVERAGE_SAMPLES_NV, &query);
		if ( query < coverageSamples)
			rb = 0;
		else if ( query > coverageSamples) 
		{
			// report back the actual number
			coverageSamples = query;
            LOGW("Warning: coverage samples is now %d\n", coverageSamples);
		}
		glGetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_COLOR_SAMPLES_NV, &query);
		if ( query < samples)
			rb = 0;
		else if ( query > samples) 
		{
			// report back the actual number
			samples = query;
            LOGW("Warning: depth-samples is now %d\n", samples);
		}
	}
	else 
	{
		// create a regular MSAA color buffer
		glRenderbufferStorageMultisample( GL_RENDERBUFFER, samples, fmt, w, h);
		// check the number of samples
		glGetRenderbufferParameteriv( GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &query);

		if ( query < samples) 
			rb = 0;
		else if ( query > samples) 
		{
			samples = query;
            LOGW("Warning: depth-samples is now %d\n", samples);
		}
	}
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
    return rb;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createRenderBufferRGBA8(int w, int h, int samples, int coverageSamples)
{
    return createRenderBuffer(w, h, samples, coverageSamples, GL_RGBA8);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createRenderBufferD24S8(int w, int h, int samples, int coverageSamples)
{
    return createRenderBuffer(w, h, samples, coverageSamples, GL_DEPTH24_STENCIL8);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
GLuint createFBO()
{
    GLuint fb;
	glGenFramebuffers(1, &fb);
    return fb;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachTexture2D(GLuint framebuffer, GLuint textureID, int colorAttachment)
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); 
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+colorAttachment, GL_TEXTURE_2D, textureID, 0);
	return CheckFramebufferStatus();
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachTexture2DMS(GLuint framebuffer, GLuint textureID, int colorAttachment)
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); 
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+colorAttachment, GL_TEXTURE_2D_MULTISAMPLE, textureID, 0);
	return CheckFramebufferStatus();
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachRenderbuffer(GLuint framebuffer, GLuint rb, int colorAttachment)
{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); 
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+colorAttachment, GL_RENDERBUFFER, rb);
	return CheckFramebufferStatus();
}
//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachDSTRenderbuffer(GLuint framebuffer, GLuint dstrb)
{
    bool bRes;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); 
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dstrb);
    bRes = CheckFramebufferStatus();
    if(!bRes) return false;
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, dstrb);
    return CheckFramebufferStatus() ;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachDSTTexture2D(GLuint framebuffer, GLuint textureDepthID, GLenum target)
{
    bool bRes;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, target, textureDepthID, 0);
    bRes = CheckFramebufferStatus();
    return bRes;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachDSTTexture2D(GLuint framebuffer, GLuint textureDepthID)
{
    return attachDSTTexture2D(framebuffer, textureDepthID, GL_TEXTURE_2D);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
bool attachDSTTexture2DMS(GLuint framebuffer, GLuint textureDepthID)
{
    return attachDSTTexture2D(framebuffer, textureDepthID, GL_TEXTURE_2D_MULTISAMPLE);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void deleteTexture(GLuint texture)
{
    glDeleteTextures(1, &texture);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void deleteRenderBuffer(GLuint rb)
{
    glDeleteRenderbuffers(1, &rb);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void deleteFBO(GLuint fbo)
{
    glDeleteFramebuffers(1, &fbo);
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void blitFBO(GLuint srcFBO, GLuint dstFBO,
    GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLenum filtering)
{
    glBindFramebuffer( GL_READ_FRAMEBUFFER, srcFBO);
    glBindFramebuffer( GL_DRAW_FRAMEBUFFER, dstFBO);
    // GL_NEAREST is needed when Stencil/depth are involved
    glBlitFramebuffer( srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT, filtering );
    glBindFramebuffer( GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0);
}
//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void blitFBONearest(GLuint srcFBO, GLuint dstFBO,
    GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1)
{
    blitFBO(srcFBO, dstFBO,srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, GL_NEAREST);
}
//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void blitFBOLinear(GLuint srcFBO, GLuint dstFBO,
    GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1)
{
    blitFBO(srcFBO, dstFBO,srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, GL_LINEAR);
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
void deleteRenderTargets()
{
    if(fboTexMS)
        deleteFBO(fboTexMS);
    if(fboTex)
        deleteFBO(fboTex);
    if(fboRbMS)
        deleteFBO(fboRbMS);
    if(fboRb)
        deleteFBO(fboRb);
    if(textureRGBA)
        deleteTexture(textureRGBA);
    if(textureRGBAMS)
        deleteTexture(textureRGBAMS);
    if(rbRGBA)
        deleteRenderBuffer(rbRGBA);
    if(rbRGBAMS)
        deleteRenderBuffer(rbRGBAMS);
    if(rbDST)
        deleteRenderBuffer(rbDST);
    if(rbDSTMS)
        deleteRenderBuffer(rbDSTMS);
    fboSz[0] = 0;
    fboSz[1] = 0;
}

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------
void buildRenderTargets(int w, int h)
{
    deleteRenderTargets();
    fboSz[0] = w;
    fboSz[1] = h;
    // a texture
    textureRGBA = createTextureRGBA8(w,h, 0,0);
    // a texture in MSAA
    textureRGBAMS = createTextureRGBA8(w,h, 8,0);
    // a renderbuffer
    rbRGBA = createRenderBufferRGBA8(w,h,0,0);
    // a renderbuffer in MSAA
    rbRGBAMS = createRenderBufferRGBA8(w,h,8,0);
    // a depth stencil
    rbDST = createRenderBufferD24S8(w,h,0,0);
    // a depth stencil in MSAA
    rbDSTMS = createRenderBufferD24S8(w,h,8,0);
    // fbo for texture MSAA as the color buffer
    fboTexMS = createFBO();
    {
        attachTexture2DMS(fboTexMS, textureRGBAMS, 0);
        attachDSTRenderbuffer(fboTexMS, rbDSTMS);
    }
    // fbo for a texture as the color buffer
    fboTex = createFBO();
    {
        attachTexture2D(fboTex, textureRGBA, 0);
        attachDSTRenderbuffer(fboTex, rbDST);
    }
    // fbo for renderbuffer MSAA as the color buffer
    fboRbMS = createFBO();
    {
        attachRenderbuffer(fboRbMS, rbRGBAMS, 0);
        attachDSTRenderbuffer(fboRbMS, rbDSTMS);
    }
    // fbo for renderbuffer as the color buffer
    fboRb = createFBO();
    {
        attachRenderbuffer(fboRb, rbRGBA, 0);
        attachDSTRenderbuffer(fboRb, rbDST);
    }

    // build a VBO for the size of the FBO
    //
    // make a VBO for Quad
    //
    if(g_vboQuad == 0)
        glGenBuffers(1, &g_vboQuad);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboQuad);
    int vertices[2*4] = { 0,0, w,0, 0,h, w,h };
    glBufferData(GL_ARRAY_BUFFER, sizeof(int)*2*4, vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
bool MyWindow::init()
{
	if(!WindowInertiaCamera::init())
		return false;

#ifdef USESVCUI
    initMFCUIBase(0, m_winSz[1]+40, m_winSz[0], 150);
#endif
#ifdef USESVCUI

    IControlCombo* pCombo = g_pWinHandler->CreateCtrlCombo("FBOMode", "FBO Mode", g_pToggleContainer);
    pCombo->AddItem("Render to Texture MS.", (size_t)RENDERTOTEXMS);
    pCombo->AddItem("Render To Texture",   (size_t)RENDERTOTEX);
    pCombo->AddItem("Render To Renderbuffer MS.",  (size_t)RENDERTORBMS);
    pCombo->AddItem("Render To Renderbuffer", (size_t)RENDERTORB);
    g_pWinHandler->VariableBind(pCombo, (int*)&fboMode);

    pCombo = g_pWinHandler->CreateCtrlCombo("BLTMode", "Blit Mode", g_pToggleContainer);
    pCombo->AddItem("Resolve with Blit", (size_t)RESOLVEWITHBLIT);
    pCombo->AddItem("Resolve with Shader&Texture Fetch", (size_t)RESOLVEWITHSHADERTEX);
    pCombo->AddItem("Resolve with Shader&Image Load", (size_t)RESOLVEWITHSHADERIMAGE);
    g_pWinHandler->VariableBind(pCombo, (int*)&blitMode);
    g_pToggleContainer->UnFold();

#endif
    //
    // easy Toggles
    //
    addToggleKeyToMFCUI(' ', &m_realtime.bNonStopRendering, "space: toggles continuous rendering\n");
    addToggleKeyToMFCUI('a', &s_bCameraAnim, "'a': animate camera\n");
    //
    // Shader compilation
    //
    if(!g_progGrid.compileProgram(g_glslv_grid, NULL, g_glslf_grid))
        return false;
    if(!g_progMesh.compileProgram(g_glslv_mesh, NULL, g_glslf_mesh))
        return false;
    if(!g_progCopyTexMSAA.compileProgram(g_glslv_Tc, NULL, g_glslf_texMSAA))
        return false;
    if(!g_progCopyTex.compileProgram(g_glslv_Tc, NULL, g_glslf_tex))
        return false;
    g_progCopyImageMSAA.compileProgram(g_glslv_Tc, NULL, g_glslf_ImageMSAA);
    g_progCopyImage.compileProgram(g_glslv_Tc, NULL, g_glslf_Image);
    //
    // Misc OGL setup
    //
    glClearColor(0.0f, 0.1f, 0.1f, 1.0f);
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    //
    // Grid floor
    //
    glGenBuffers(1, &g_vboGrid);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboGrid);
    #define GRIDDEF 20
    #define GRIDSZ 1.0
    vec3f *data = new vec3f[GRIDDEF*4];
    vec3f *p = data;
    int j=0;
    for(int i=0; i<GRIDDEF; i++)
    {
        *(p++) = vec3f(-GRIDSZ, 0.0, GRIDSZ*(-1.0f+2.0f*(float)i/(float)GRIDDEF));
        *(p++) = vec3f( GRIDSZ*(1.0f-2.0f/(float)GRIDDEF), 0.0, GRIDSZ*(-1.0f+2.0f*(float)i/(float)GRIDDEF));
        *(p++) = vec3f(GRIDSZ*(-1.0f+2.0f*(float)i/(float)GRIDDEF), 0.0, -GRIDSZ);
        *(p++) = vec3f(GRIDSZ*(-1.0f+2.0f*(float)i/(float)GRIDDEF), 0.0, GRIDSZ*(1.0f-2.0f/(float)GRIDDEF));
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(vec3f)*GRIDDEF*4, data[0].vec_array, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //
    // 3D Model
    //
    LOGI("Loading Mesh..." MODELNAME "\n");
    if(!(meshFile = bk3d::load(MODELNAME)))
        if(!(meshFile = bk3d::load(PROJECT_RELDIRECTORY MODELNAME)))
            meshFile = bk3d::load(PROJECT_ABSDIRECTORY MODELNAME);
    if(meshFile)
    {
        // create VBOs
	    for(int i=0; i< meshFile->pMeshes->n; i++)
	    {
		    bk3d::Mesh *pMesh = meshFile->pMeshes->p[i];
            for(int s=0; s<pMesh->pSlots->n; s++)
            {
                bk3d::Slot* pS = pMesh->pSlots->p[s];
		        glGenBuffers(1, (GLuint*)&pS->userData); // store directly to a user-space dedicated to this kind of things
                glBindBuffer(GL_ARRAY_BUFFER, pS->userData);
                glBufferData(GL_ARRAY_BUFFER, pS->vtxBufferSizeBytes, pS->pVtxBufferData, GL_STATIC_DRAW);
            }
            for(int pg=0; pg<pMesh->pPrimGroups->n; pg++)
            {
                bk3d::PrimGroup* pPG = pMesh->pPrimGroups->p[pg];
                glGenBuffers(1, (GLuint*)&pPG->userPtr);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (GLuint)pMesh->pPrimGroups->p[pg]->userPtr);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, pPG->indexArrayByteSize, pPG->pIndexBufferData, GL_STATIC_DRAW);
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	    }
	    //
	    // Some adjustment for the display
	    //
	    float min[3] = {1000.0, 1000.0, 1000.0};
	    float max[3] = {-1000.0, -1000.0, -1000.0};
	    for(int i=0; i<meshFile->pMeshes->n; i++)
	    {
		    bk3d::Mesh *pMesh = meshFile->pMeshes->p[i];
		    if(pMesh->aabbox.min[0] < min[0]) min[0] = pMesh->aabbox.min[0];
		    if(pMesh->aabbox.min[1] < min[1]) min[1] = pMesh->aabbox.min[1];
		    if(pMesh->aabbox.min[2] < min[2]) min[2] = pMesh->aabbox.min[2];
		    if(pMesh->aabbox.max[0] > max[0]) max[0] = pMesh->aabbox.max[0];
		    if(pMesh->aabbox.max[1] > max[1]) max[1] = pMesh->aabbox.max[1];
		    if(pMesh->aabbox.max[2] > max[2]) max[2] = pMesh->aabbox.max[2];
	    }
	    g_posOffset[0] = (max[0] + min[0])*0.5f;
	    g_posOffset[1] = (max[1] + min[1])*0.5f;
	    g_posOffset[2] = (max[2] + min[2])*0.5f;
	    float bigger = 0;
	    if((max[0]-min[0]) > bigger) bigger = (max[0]-min[0]);
	    if((max[1]-min[1]) > bigger) bigger = (max[1]-min[1]);
	    if((max[2]-min[2]) > bigger) bigger = (max[2]-min[2]);
	    if((bigger) > 0.001)
	    {
		    g_scale = 1.0 / bigger;
		    PRINTF(("Scaling the model by %f...\n", g_scale));
	    }
    } else {
        LOGE("error in loading mesh\n");
    }
    // --------------------------------------------
    // FBOs
    //
    buildRenderTargets(m_winSz[0], m_winSz[1]);

	m_validated = true;
    return true;
}
//------------------------------------------------------------------------------
void MyWindow::shutdown()
{
#ifdef USESVCUI
    shutdownMFCUI();
#endif
}

//------------------------------------------------------------------------------
void MyWindow::reshape(int w, int h)
{
	WindowInertiaCamera::reshape(w, h);
    //
    // rebuild the FBOs to match the new size
    //
    if(m_validated)
        buildRenderTargets(w, h);
}

//------------------------------------------------------------------------------
#define KEYTAU 0.10f
void MyWindow::keyboard(NVPWindow::KeyCode key, MyWindow::ButtonAction action, int mods, int x, int y)
{
	WindowInertiaCamera::keyboard(key, action, mods, x, y);
	if(action == MyWindow::BUTTON_RELEASE)
        return;
    switch(key)
    {
    case NVPWindow::KEY_F1:
        break;
	//...
    case NVPWindow::KEY_F12:
        break;
    }
#ifdef USESVCUI
    flushMFCUIToggle(key);
#endif
}
//------------------------------------------------------------------------------
void MyWindow::keyboardchar(unsigned char key, int mods, int x, int y)
{
    WindowInertiaCamera::keyboardchar(key,  mods, x, y);
    switch( key )
    {
        case '1':
            fboMode = RENDERTOTEXMS;
            LOGI("Rendering to FBO made of a color texture MSAA\n");
            break;
        case '2':
            fboMode = RENDERTOTEX;
            LOGI("Rendering to FBO made of a color texture\n");
            break;
        case '3':
            fboMode = RENDERTORBMS;
            LOGI("Rendering to FBO made of a color renderbuffer MSAA\n");
            break;
        case '4':
            fboMode = RENDERTORB;
            LOGI("Rendering to FBO made of a renderbuffer\n");
            break;
        case '5':
            blitMode = RESOLVEWITHBLIT;
            LOGI("blitting using framebufferblit()\n");
            break;
        case '6':
            blitMode = RESOLVEWITHSHADERTEX;
            LOGI("blitting using fullscreenquad and texture\n");
            break;
        case '7':
            blitMode = RESOLVEWITHSHADERIMAGE;
            LOGI("blitting using fullscreenquad and image\n");
            break;
        default:
            break;
    }
#ifdef USESVCUI
    g_pWinHandler->VariableFlush(&fboMode);
    g_pWinHandler->VariableFlush(&blitMode);
    flushMFCUIToggle(key);
#endif
}

void MyWindow::renderScene()
{
    /////////////////////////////////////////////////
    //// Grid floor
    g_progGrid.enable();
    mat4f mWVP;
    mWVP = m_projection * m_camera.m4_view /* * World transf...*/;
    g_progGrid.setUniformMatrix4fv("mWVP", mWVP.mat_array, false);
    g_progGrid.setUniform3f("diffuse", 0.3, 0.3, 1.0);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboGrid);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3f), NULL);
    glDrawArrays(GL_LINES, 0, GRIDDEF*4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    //g_progGrid.disable();
    ////////////////////////////////////////////////////////////////////////////////////
    // Display Meshes
    // Note that we keep it too simple here: assuming pos + normals are at attr 0 & 1
	//
    if(meshFile)
    {
        g_progMesh.enable();
        vec3f lightDir(0.4,0.8,0.3);
        lightDir.normalize();
        g_progMesh.setUniform3f("lightDir", lightDir[0], lightDir[1], lightDir[2]);
        mWVP.rotate(nv_to_rad*180.0, vec3f(0,1,0));
        mWVP.scale(g_scale);
	    mWVP.translate(-g_posOffset);
        g_progMesh.setUniformMatrix4fv("mWVP", mWVP.mat_array, false);
	    glEnableVertexAttribArray(0);
	    glEnableVertexAttribArray(1);
	    for(int i=0; i< meshFile->pMeshes->n; i++)
	    {
		    bk3d::Mesh *pMesh = meshFile->pMeshes->p[i];

            bk3d::Attribute* pAttrPos = pMesh->pAttributes->p[0];
		    glBindBuffer(GL_ARRAY_BUFFER, pMesh->pSlots->p[pAttrPos->slot]->userData);
		    glVertexAttribPointer(0,
			    pAttrPos->numComp, 
			    pAttrPos->formatGL,
                GL_FALSE,
			    pAttrPos->strideBytes,
			    (void*)pAttrPos->dataOffsetBytes);

            bk3d::Attribute* pAttrN = pMesh->pAttributes->p[1];
            if(pAttrN->slot != pAttrPos->slot)
                glBindBuffer(GL_ARRAY_BUFFER, pMesh->pSlots->p[pAttrN->slot]->userData);
		    glVertexAttribPointer(1, pAttrN->numComp,
			    pAttrN->formatGL,
                GL_TRUE,
			    pAttrN->strideBytes,
			    (void*)pAttrN->dataOffsetBytes);
		    for(int pg=0; pg<pMesh->pPrimGroups->n; pg++)
		    {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, (GLuint)pMesh->pPrimGroups->p[pg]->userPtr);
			    bk3d::Material *pMat = pMesh->pPrimGroups->p[pg]->pMaterial;
			    if(pMat)// && g_bUseMaterial)
                    g_progMesh.setUniform3f("diffuse", pMat->MaterialData().diffuse[0], pMat->MaterialData().diffuse[1], pMat->MaterialData().diffuse[2]);
			    else
				    g_progMesh.setUniform3f("diffuse", 0.8, 0.8, 0.8);
			    glDrawElements(
				    pMesh->pPrimGroups->p[pg]->topologyGL,
				    pMesh->pPrimGroups->p[pg]->indexCount,
				    pMesh->pPrimGroups->p[pg]->indexFormatGL,
				    NULL);
		    }
	    }
	    glDisableVertexAttribArray(0);
	    glDisableVertexAttribArray(1);
    }
}

void MyWindow::display()
{
    NXPROFILEFUNC(__FUNCTION__);
    WindowInertiaCamera::display();
    //
    // Simple camera change for animation
    //
    if(s_bCameraAnim)
    {
      float dt = (float)m_realtime.getTiming();
      s_cameraAnimIntervals -= dt;
      if(s_cameraAnimIntervals <= 0.0)
      {
          s_cameraAnimIntervals = ANIMINTERVALL;
          m_camera.look_at(s_cameraAnim[s_cameraAnimItem].eye, s_cameraAnim[s_cameraAnimItem].focus);
          s_cameraAnimItem++;
          if(s_cameraAnimItem >= s_cameraAnimItems)
              s_cameraAnimItem = 0;
      }
    }

    GLuint fbo;
    switch(fboMode)
    {
    case RENDERTOTEXMS:
        fbo = fboTexMS;
        break;
    case RENDERTOTEX:
        fbo = fboTex;
        break;
    case RENDERTORBMS:
        fbo = fboRbMS;
        break;
    case RENDERTORB:
        fbo = fboRb;
        break;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    {
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        renderScene();
    }
    // Done. Back to the backbuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    switch(blitMode)
    {
    case RESOLVEWITHBLIT:
        blitFBONearest(fbo, 0,
            0, 0, m_winSz[0], m_winSz[1], 0, 0, m_winSz[0], m_winSz[1]);
        break;
    case RESOLVEWITHSHADERTEX:
        if(fboMode == RENDERTOTEXMS)
        {
            g_progCopyTexMSAA.enable();
            g_progCopyTexMSAA.setUniform2i("viewportSz", m_winSz[0], m_winSz[1]);
            g_progCopyTexMSAA.bindTexture("samplerMS", textureRGBAMS, GL_TEXTURE_2D_MULTISAMPLE, 0);
            glBindBuffer(GL_ARRAY_BUFFER, g_vboQuad);
            glEnableVertexAttribArray(0);
            glVertexAttribIPointer(0, 2, GL_INT, sizeof(int)*2, NULL);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableVertexAttribArray(0);
        }
        else if(fboMode == RENDERTOTEX)
        {
            g_progCopyTex.enable();
            g_progCopyTex.setUniform2i("viewportSz", m_winSz[0], m_winSz[1]);
            g_progCopyTex.bindTexture("s", textureRGBA, GL_TEXTURE_2D, 0);
            glBindBuffer(GL_ARRAY_BUFFER, g_vboQuad);
            glEnableVertexAttribArray(0);
            glVertexAttribIPointer(0, 2, GL_INT, sizeof(int)*2, NULL);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableVertexAttribArray(0);
        }
        break;
    case RESOLVEWITHSHADERIMAGE:
        if((fboMode == RENDERTOTEXMS)&&(g_progCopyImageMSAA.getProgId()))
        {
            g_progCopyImageMSAA.enable();
            g_progCopyImageMSAA.setUniform2i("viewportSz", m_winSz[0], m_winSz[1]);
            g_progCopyImageMSAA.bindImage("imageMS", 0, textureRGBAMS, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
            glBindBuffer(GL_ARRAY_BUFFER, g_vboQuad);
            glEnableVertexAttribArray(0);
            glVertexAttribIPointer(0, 2, GL_INT, sizeof(int)*2, NULL);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableVertexAttribArray(0);
        }
        else if((fboMode == RENDERTOTEX)&&(g_progCopyImage.getProgId()))
        {
            g_progCopyImage.enable();
            g_progCopyImage.setUniform2i("viewportSz", m_winSz[0], m_winSz[1]);
            g_progCopyImage.bindImage("image", 0, textureRGBA, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
            glBindBuffer(GL_ARRAY_BUFFER, g_vboQuad);
            glEnableVertexAttribArray(0);
            glVertexAttribIPointer(0, 2, GL_INT, sizeof(int)*2, NULL);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableVertexAttribArray(0);
        }
        break;
    }

    ///////////////////////////////////////////////
    // additional HUD stuff
	WindowInertiaCamera::displayHUD();

    swapBuffers();
}
/////////////////////////////////////////////////////////////////////////
// Main initialization point
//
int sample_main(int argc, const char** argv)
{
    // you can create more than only one
    static MyWindow myWindow;

    NVPWindow::ContextFlags context(
    4,      //major;
    3,      //minor;
    true,   //core;
    8,      //MSAA;
    24,     //depth bits
    8,      //stencil bits
    true,   //debug;
    false,  //robust;
    false,  //forward;
    NULL   //share;
    );

    if(!myWindow.create("Simple FBO", &context))
        return false;

    myWindow.makeContextCurrent();
    myWindow.swapInterval(0);

    while(MyWindow::sysPollEvents(false) )
    {
        myWindow.idle();
    }
    return true;
}
