// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/modules/face_geometry/libs/effect_renderer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/shader_util.h"
#include "mediapipe/modules/face_geometry/libs/mesh_3d_utils.h"
#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"

namespace mediapipe::face_geometry {
namespace {

struct RenderableMesh3d {
  static mediapipe::StatusOr<RenderableMesh3d> CreateFromProtoMesh3d(
      const Mesh3d& proto_mesh_3d) {
    Mesh3d::VertexType vertex_type = proto_mesh_3d.vertex_type();

    RenderableMesh3d renderable_mesh_3d;
    renderable_mesh_3d.vertex_size = GetVertexSize(vertex_type);
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.vertex_position_size,
        GetVertexComponentSize(vertex_type, VertexComponent::POSITION),
        _ << "Failed to get the position vertex size!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.tex_coord_position_size,
        GetVertexComponentSize(vertex_type, VertexComponent::TEX_COORD),
        _ << "Failed to get the tex coord vertex size!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.vertex_position_offset,
        GetVertexComponentOffset(vertex_type, VertexComponent::POSITION),
        _ << "Failed to get the position vertex offset!");
    ASSIGN_OR_RETURN(
        renderable_mesh_3d.tex_coord_position_offset,
        GetVertexComponentOffset(vertex_type, VertexComponent::TEX_COORD),
        _ << "Failed to get the tex coord vertex offset!");

    switch (proto_mesh_3d.primitive_type()) {
      case Mesh3d::TRIANGLE:
        renderable_mesh_3d.primitive_type = GL_TRIANGLES;
        break;

      default:
        RET_CHECK_FAIL() << "Only triangle primitive types are supported!";
    }

    renderable_mesh_3d.vertex_buffer.reserve(
        proto_mesh_3d.vertex_buffer_size());
    for (float vertex_element : proto_mesh_3d.vertex_buffer()) {
      renderable_mesh_3d.vertex_buffer.push_back(vertex_element);
    }

    renderable_mesh_3d.index_buffer.reserve(proto_mesh_3d.index_buffer_size());
    for (uint32_t index_element : proto_mesh_3d.index_buffer()) {
      RET_CHECK_LE(index_element, std::numeric_limits<uint16_t>::max())
          << "Index buffer elements must fit into the `uint16` type in order "
             "to be renderable!";

      renderable_mesh_3d.index_buffer.push_back(
          static_cast<uint16_t>(index_element));
    }

    return renderable_mesh_3d;
  }

  uint32_t vertex_size;
  uint32_t vertex_position_size;
  uint32_t tex_coord_position_size;
  uint32_t vertex_position_offset;
  uint32_t tex_coord_position_offset;
  uint32_t primitive_type;

  std::vector<float> vertex_buffer;
  std::vector<uint16_t> index_buffer;
};

class Texture {
 public:
  static mediapipe::StatusOr<std::unique_ptr<Texture>> WrapExternalTexture(
      GLuint handle, GLenum target, int width, int height) {
    RET_CHECK(handle) << "External texture must have a non-null handle!";
    return absl::WrapUnique(new Texture(handle, target, width, height,
                                        /*is_owned*/ false));
  }

  static mediapipe::StatusOr<std::unique_ptr<Texture>> CreateFromImageFrame(
      const ImageFrame& image_frame) {
    RET_CHECK(image_frame.IsAligned(ImageFrame::kGlDefaultAlignmentBoundary))
        << "Image frame memory must be aligned for GL usage!";

    RET_CHECK(image_frame.Width() > 0 && image_frame.Height() > 0)
        << "Image frame must have positive dimensions!";

    RET_CHECK(image_frame.Format() == ImageFormat::SRGB ||
              image_frame.Format() == ImageFormat::SRGBA)
        << "Image frame format must be either SRGB or SRGBA!";

    GLint image_format;
    switch (image_frame.NumberOfChannels()) {
      case 3:
        image_format = GL_RGB;
        break;
      case 4:
        image_format = GL_RGBA;
        break;
      default:
        RET_CHECK_FAIL()
            << "Unexpected number of channels; expected 3 or 4, got "
            << image_frame.NumberOfChannels() << "!";
    }

    GLuint handle;
    glGenTextures(1, &handle);
    RET_CHECK(handle) << "Failed to initialize an OpenGL texture!";

    glBindTexture(GL_TEXTURE_2D, handle);
    glTexParameteri(GL_TEXTURE_2D, GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, image_format, image_frame.Width(),
                 image_frame.Height(), 0, image_format, GL_UNSIGNED_BYTE,
                 image_frame.PixelData());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    return absl::WrapUnique(new Texture(
        handle, GL_TEXTURE_2D, image_frame.Width(), image_frame.Height(),
        /*is_owned*/ true));
  }

  ~Texture() {
    if (is_owned_) {
      glDeleteProgram(handle_);
    }
  }

  GLuint handle() const { return handle_; }
  GLenum target() const { return target_; }
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  Texture(GLuint handle, GLenum target, int width, int height, bool is_owned)
      : handle_(handle),
        target_(target),
        width_(width),
        height_(height),
        is_owned_(is_owned) {}

  GLuint handle_;
  GLenum target_;
  int width_;
  int height_;
  bool is_owned_;
};

class RenderTarget {
 public:
  static mediapipe::StatusOr<std::unique_ptr<RenderTarget>> Create() {
    GLuint framebuffer_handle;
    glGenFramebuffers(1, &framebuffer_handle);
    RET_CHECK(framebuffer_handle)
        << "Failed to initialize an OpenGL framebuffer!";

    return absl::WrapUnique(new RenderTarget(framebuffer_handle));
  }

  ~RenderTarget() {
    glDeleteFramebuffers(1, &framebuffer_handle_);
    // Renderbuffer handle might have never been created if this render target
    // is destroyed before `SetColorbuffer()` is called for the first time.
    if (renderbuffer_handle_) {
      glDeleteFramebuffers(1, &renderbuffer_handle_);
    }
  }

  mediapipe::Status SetColorbuffer(const Texture& colorbuffer_texture) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_handle_);
    glViewport(0, 0, colorbuffer_texture.width(), colorbuffer_texture.height());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(colorbuffer_texture.target(), colorbuffer_texture.handle());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           colorbuffer_texture.target(),
                           colorbuffer_texture.handle(),
                           /*level*/ 0);
    glBindTexture(colorbuffer_texture.target(), 0);

    // If the existing depth buffer has different dimensions, delete it.
    if (renderbuffer_handle_ &&
        (viewport_width_ != colorbuffer_texture.width() ||
         viewport_height_ != colorbuffer_texture.height())) {
      glDeleteRenderbuffers(1, &renderbuffer_handle_);
      renderbuffer_handle_ = 0;
    }

    // If there is no depth buffer, create one.
    if (!renderbuffer_handle_) {
      glGenRenderbuffers(1, &renderbuffer_handle_);
      RET_CHECK(renderbuffer_handle_)
          << "Failed to initialize an OpenGL renderbuffer!";
      glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer_handle_);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,
                            colorbuffer_texture.width(),
                            colorbuffer_texture.height());
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                GL_RENDERBUFFER, renderbuffer_handle_);
      glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    viewport_width_ = colorbuffer_texture.width();
    viewport_height_ = colorbuffer_texture.height();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glFlush();

    return mediapipe::OkStatus();
  }

  void Bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_handle_);
    glViewport(0, 0, viewport_width_, viewport_height_);
  }

  void Unbind() const { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

  void Clear() const {
    Bind();
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClearDepthf(1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);

    Unbind();
    glFlush();
  }

 private:
  explicit RenderTarget(GLuint framebuffer_handle)
      : framebuffer_handle_(framebuffer_handle),
        renderbuffer_handle_(0),
        viewport_width_(-1),
        viewport_height_(-1) {}

  GLuint framebuffer_handle_;
  GLuint renderbuffer_handle_;
  int viewport_width_;
  int viewport_height_;
};

class Renderer {
 public:
  enum class RenderMode { OPAQUE, OVERDRAW, OCCLUSION };

  static mediapipe::StatusOr<std::unique_ptr<Renderer>> Create() {
    static const GLint kAttrLocation[NUM_ATTRIBUTES] = {
        ATTRIB_VERTEX,
        ATTRIB_TEXTURE_POSITION,
    };
    static const GLchar* kAttrName[NUM_ATTRIBUTES] = {
        "position",
        "tex_coord",
    };

    static const GLchar* kVertSrc = R"(
      uniform mat4 projection_mat;
      uniform mat4 model_mat;

      attribute vec4 position;
      attribute vec4 tex_coord;

      varying vec2 v_tex_coord;

      void main() {
        v_tex_coord = tex_coord.xy;
        gl_Position = projection_mat * model_mat * position;
      }
    )";

    static const GLchar* kFragSrc = R"(
      precision mediump float;

      varying vec2 v_tex_coord;
      uniform sampler2D texture;

      void main() {
        gl_FragColor = texture2D(texture, v_tex_coord);
      }
    )";

    GLuint program_handle = 0;
    GlhCreateProgram(kVertSrc, kFragSrc, NUM_ATTRIBUTES,
                     (const GLchar**)&kAttrName[0], kAttrLocation,
                     &program_handle);
    RET_CHECK(program_handle) << "Problem initializing the texture program!";
    GLint projection_mat_uniform =
        glGetUniformLocation(program_handle, "projection_mat");
    GLint model_mat_uniform = glGetUniformLocation(program_handle, "model_mat");
    GLint texture_uniform = glGetUniformLocation(program_handle, "texture");

    RET_CHECK_NE(projection_mat_uniform, -1)
        << "Failed to find `projection_mat` uniform!";
    RET_CHECK_NE(model_mat_uniform, -1)
        << "Failed to find `model_mat` uniform!";
    RET_CHECK_NE(texture_uniform, -1) << "Failed to find `texture` uniform!";

    return absl::WrapUnique(new Renderer(program_handle, projection_mat_uniform,
                                         model_mat_uniform, texture_uniform));
  }

  ~Renderer() { glDeleteProgram(program_handle_); }

  mediapipe::Status Render(const RenderTarget& render_target,
                           const Texture& texture,
                           const RenderableMesh3d& mesh_3d,
                           const std::array<float, 16>& projection_mat,
                           const std::array<float, 16>& model_mat,
                           RenderMode render_mode) const {
    glUseProgram(program_handle_);
    // Set up the GL state.
    glEnable(GL_BLEND);
    glFrontFace(GL_CCW);
    switch (render_mode) {
      case RenderMode::OPAQUE:
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        break;

      case RenderMode::OVERDRAW:
        glBlendFunc(GL_ONE, GL_ZERO);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        break;

      case RenderMode::OCCLUSION:
        glBlendFunc(GL_ZERO, GL_ONE);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        break;
    }

    render_target.Bind();
    // Set up vertex attributes.
    glVertexAttribPointer(
        ATTRIB_VERTEX, mesh_3d.vertex_position_size, GL_FLOAT, 0,
        mesh_3d.vertex_size * sizeof(float),
        mesh_3d.vertex_buffer.data() + mesh_3d.vertex_position_offset);
    glEnableVertexAttribArray(ATTRIB_VERTEX);
    glVertexAttribPointer(
        ATTRIB_TEXTURE_POSITION, mesh_3d.tex_coord_position_size, GL_FLOAT, 0,
        mesh_3d.vertex_size * sizeof(float),
        mesh_3d.vertex_buffer.data() + mesh_3d.tex_coord_position_offset);
    glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
    // Set up textures and uniforms.
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(texture.target(), texture.handle());
    glUniform1i(texture_uniform_, 1);
    glUniformMatrix4fv(projection_mat_uniform_, 1, GL_FALSE,
                       projection_mat.data());
    glUniformMatrix4fv(model_mat_uniform_, 1, GL_FALSE, model_mat.data());
    // Draw the mesh.
    glDrawElements(mesh_3d.primitive_type, mesh_3d.index_buffer.size(),
                   GL_UNSIGNED_SHORT, mesh_3d.index_buffer.data());
    // Unbind textures and uniforms.
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(texture.target(), 0);
    render_target.Unbind();
    // Unbind vertex attributes.
    glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
    glDisableVertexAttribArray(ATTRIB_VERTEX);
    // Restore the GL state.
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glUseProgram(0);
    glFlush();

    return mediapipe::OkStatus();
  }

 private:
  enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

  Renderer(GLuint program_handle, GLint projection_mat_uniform,
           GLint model_mat_uniform, GLint texture_uniform)
      : program_handle_(program_handle),
        projection_mat_uniform_(projection_mat_uniform),
        model_mat_uniform_(model_mat_uniform),
        texture_uniform_(texture_uniform) {}

  GLuint program_handle_;
  GLint projection_mat_uniform_;
  GLint model_mat_uniform_;
  GLint texture_uniform_;
};

class EffectRendererImpl : public EffectRenderer {
 public:
  EffectRendererImpl(
      const Environment& environment,
      std::unique_ptr<RenderTarget> render_target,
      std::unique_ptr<Renderer> renderer,
      RenderableMesh3d&& renderable_quad_mesh_3d,
      absl::optional<RenderableMesh3d>&& renderable_effect_mesh_3d,
      std::unique_ptr<Texture> empty_color_texture,
      std::unique_ptr<Texture> effect_texture)
      : environment_(environment),
        render_target_(std::move(render_target)),
        renderer_(std::move(renderer)),
        renderable_quad_mesh_3d_(std::move(renderable_quad_mesh_3d)),
        renderable_effect_mesh_3d_(std::move(renderable_effect_mesh_3d)),
        empty_color_texture_(std::move(empty_color_texture)),
        effect_texture_(std::move(effect_texture)),
        identity_matrix_(Create4x4IdentityMatrix()) {}

  mediapipe::Status RenderEffect(
      const std::vector<FaceGeometry>& multi_face_geometry,
      int frame_width,            //
      int frame_height,           //
      GLenum src_texture_target,  //
      GLuint src_texture_name,    //
      GLenum dst_texture_target,  //
      GLuint dst_texture_name) {
    // Validate input arguments.
    MP_RETURN_IF_ERROR(ValidateFrameDimensions(frame_width, frame_height))
        << "Invalid frame dimensions!";
    RET_CHECK(src_texture_name > 0 && dst_texture_name > 0)
        << "Both source and destination texture names must be non-null!";
    RET_CHECK_NE(src_texture_name, dst_texture_name)
        << "Source and destination texture names must be different!";

    // Validate all input face geometries.
    for (const FaceGeometry& face_geometry : multi_face_geometry) {
      MP_RETURN_IF_ERROR(ValidateFaceGeometry(face_geometry))
          << "Invalid face geometry!";
    }

    // Wrap both source and destination textures.
    ASSIGN_OR_RETURN(
        std::unique_ptr<Texture> src_texture,
        Texture::WrapExternalTexture(src_texture_name, src_texture_target,
                                     frame_width, frame_height),
        _ << "Failed to wrap the external source texture");
    ASSIGN_OR_RETURN(
        std::unique_ptr<Texture> dst_texture,
        Texture::WrapExternalTexture(dst_texture_name, dst_texture_target,
                                     frame_width, frame_height),
        _ << "Failed to wrap the external destination texture");

    // Set the destination texture as the color buffer. Then, clear both the
    // color and the depth buffers for the render target.
    MP_RETURN_IF_ERROR(render_target_->SetColorbuffer(*dst_texture))
        << "Failed to set the destination texture as the colorbuffer!";
    render_target_->Clear();

    // Render the source texture on top of the quad mesh (i.e. make a copy)
    // into the render target.
    MP_RETURN_IF_ERROR(renderer_->Render(
        *render_target_, *src_texture, renderable_quad_mesh_3d_,
        identity_matrix_, identity_matrix_, Renderer::RenderMode::OVERDRAW))
        << "Failed to render the source texture on top of the quad mesh!";

    // Extract pose transform matrices and meshes from the face geometry data;
    const int num_faces = multi_face_geometry.size();

    std::vector<std::array<float, 16>> face_pose_transform_matrices(num_faces);
    std::vector<RenderableMesh3d> renderable_face_meshes(num_faces);
    for (int i = 0; i < num_faces; ++i) {
      const FaceGeometry& face_geometry = multi_face_geometry[i];

      // Extract the face pose transformation matrix.
      ASSIGN_OR_RETURN(
          face_pose_transform_matrices[i],
          Convert4x4MatrixDataToArrayFormat(
              face_geometry.pose_transform_matrix()),
          _ << "Failed to extract the face pose transformation matrix!");

      // Extract the face mesh as a renderable.
      ASSIGN_OR_RETURN(
          renderable_face_meshes[i],
          RenderableMesh3d::CreateFromProtoMesh3d(face_geometry.mesh()),
          _ << "Failed to extract a renderable face mesh!");
    }

    // Create a perspective matrix using the frame aspect ratio.
    std::array<float, 16> perspective_matrix = CreatePerspectiveMatrix(
        /*aspect_ratio*/ static_cast<float>(frame_width) / frame_height);

    // Render a face mesh occluder for each face.
    for (int i = 0; i < num_faces; ++i) {
      const std::array<float, 16>& face_pose_transform_matrix =
          face_pose_transform_matrices[i];
      const RenderableMesh3d& renderable_face_mesh = renderable_face_meshes[i];

      // Render the face mesh using the empty color texture, i.e. the face
      // mesh occluder.
      //
      // For occlusion, the pose transformation is moved ~1mm away from camera
      // in order to allow the face mesh texture to be rendered without
      // failing the depth test.
      std::array<float, 16> occlusion_face_pose_transform_matrix =
          face_pose_transform_matrix;
      occlusion_face_pose_transform_matrix[14] -= 0.1f;  // ~ 1mm
      MP_RETURN_IF_ERROR(renderer_->Render(
          *render_target_, *empty_color_texture_, renderable_face_mesh,
          perspective_matrix, occlusion_face_pose_transform_matrix,
          Renderer::RenderMode::OCCLUSION))
          << "Failed to render the face mesh occluder!";
    }

    // Render the main face mesh effect component for each face.
    for (int i = 0; i < num_faces; ++i) {
      const std::array<float, 16>& face_pose_transform_matrix =
          face_pose_transform_matrices[i];

      // If there is no effect 3D mesh provided, then the face mesh itself is
      // used as a topology for rendering (for example, this can be used for
      // facepaint effects or AR makeup).
      const RenderableMesh3d& main_effect_mesh_3d =
          renderable_effect_mesh_3d_ ? *renderable_effect_mesh_3d_
                                     : renderable_face_meshes[i];

      MP_RETURN_IF_ERROR(renderer_->Render(
          *render_target_, *effect_texture_, main_effect_mesh_3d,
          perspective_matrix, face_pose_transform_matrix,
          Renderer::RenderMode::OPAQUE))
          << "Failed to render the main effect pass!";
    }

    // At this point in the code, the destination texture must contain the
    // correctly renderer effect, so we should just return.
    return mediapipe::OkStatus();
  }

 private:
  std::array<float, 16> CreatePerspectiveMatrix(float aspect_ratio) const {
    static constexpr float kDegreesToRadians = M_PI / 180.f;

    std::array<float, 16> perspective_matrix;
    perspective_matrix.fill(0.f);

    const auto& env_camera = environment_.perspective_camera();
    // Standard perspective projection matrix calculations.
    const float f = 1.0f / std::tan(kDegreesToRadians *
                                    env_camera.vertical_fov_degrees() / 2.f);

    const float denom = 1.0f / (env_camera.near() - env_camera.far());
    perspective_matrix[0] = f / aspect_ratio;
    perspective_matrix[5] = f;
    perspective_matrix[10] = (env_camera.near() + env_camera.far()) * denom;
    perspective_matrix[11] = -1.f;
    perspective_matrix[14] = 2.f * env_camera.far() * env_camera.near() * denom;

    // If the environment's origin point location is in the top left corner,
    // then skip additional flip along Y-axis is required to render correctly.
    if (environment_.origin_point_location() ==
        OriginPointLocation::TOP_LEFT_CORNER) {
      perspective_matrix[5] *= -1.f;
    }

    return perspective_matrix;
  }

  static std::array<float, 16> Create4x4IdentityMatrix() {
    return {1.f, 0.f, 0.f, 0.f,  //
            0.f, 1.f, 0.f, 0.f,  //
            0.f, 0.f, 1.f, 0.f,  //
            0.f, 0.f, 0.f, 1.f};
  }

  static mediapipe::StatusOr<std::array<float, 16>>
  Convert4x4MatrixDataToArrayFormat(const MatrixData& matrix_data) {
    RET_CHECK(matrix_data.rows() == 4 &&  //
              matrix_data.cols() == 4 &&  //
              matrix_data.packed_data_size() == 16)
        << "The matrix data must define a 4x4 matrix!";

    std::array<float, 16> matrix_array;
    for (int i = 0; i < 16; i++) {
      matrix_array[i] = matrix_data.packed_data(i);
    }

    // Matrix array must be in the OpenGL-friendly column-major order. If
    // `matrix_data` is in the row-major order, then transpose.
    if (matrix_data.layout() == MatrixData::ROW_MAJOR) {
      std::swap(matrix_array[1], matrix_array[4]);
      std::swap(matrix_array[2], matrix_array[8]);
      std::swap(matrix_array[3], matrix_array[12]);
      std::swap(matrix_array[6], matrix_array[9]);
      std::swap(matrix_array[7], matrix_array[13]);
      std::swap(matrix_array[11], matrix_array[14]);
    }

    return matrix_array;
  }

  Environment environment_;

  std::unique_ptr<RenderTarget> render_target_;
  std::unique_ptr<Renderer> renderer_;

  RenderableMesh3d renderable_quad_mesh_3d_;
  absl::optional<RenderableMesh3d> renderable_effect_mesh_3d_;

  std::unique_ptr<Texture> empty_color_texture_;
  std::unique_ptr<Texture> effect_texture_;

  std::array<float, 16> identity_matrix_;
};

Mesh3d CreateQuadMesh3d() {
  static constexpr float kQuadMesh3dVertexBuffer[] = {
      -1.f, -1.f, 0.f, 0.f, 0.f,  //
      1.f,  -1.f, 0.f, 1.f, 0.f,  //
      -1.f, 1.f,  0.f, 0.f, 1.f,  //
      1.f,  1.f,  0.f, 1.f, 1.f,  //
  };
  static constexpr uint16_t kQuadMesh3dIndexBuffer[] = {0, 1, 2, 1, 3, 2};

  static constexpr int kQuadMesh3dVertexBufferSize =
      sizeof(kQuadMesh3dVertexBuffer) / sizeof(float);
  static constexpr int kQuadMesh3dIndexBufferSize =
      sizeof(kQuadMesh3dIndexBuffer) / sizeof(uint16_t);

  Mesh3d quad_mesh_3d;
  quad_mesh_3d.set_vertex_type(Mesh3d::VERTEX_PT);
  quad_mesh_3d.set_primitive_type(Mesh3d::TRIANGLE);
  for (int i = 0; i < kQuadMesh3dVertexBufferSize; ++i) {
    quad_mesh_3d.add_vertex_buffer(kQuadMesh3dVertexBuffer[i]);
  }
  for (int i = 0; i < kQuadMesh3dIndexBufferSize; ++i) {
    quad_mesh_3d.add_index_buffer(kQuadMesh3dIndexBuffer[i]);
  }

  return quad_mesh_3d;
}

ImageFrame CreateEmptyColorTexture() {
  static constexpr ImageFormat::Format kEmptyColorTextureFormat =
      ImageFormat::SRGBA;
  static constexpr int kEmptyColorTextureWidth = 1;
  static constexpr int kEmptyColorTextureHeight = 1;

  ImageFrame empty_color_texture(
      kEmptyColorTextureFormat, kEmptyColorTextureWidth,
      kEmptyColorTextureHeight, ImageFrame::kGlDefaultAlignmentBoundary);
  empty_color_texture.SetToZero();

  return empty_color_texture;
}

}  // namespace

mediapipe::StatusOr<std::unique_ptr<EffectRenderer>> CreateEffectRenderer(
    const Environment& environment,                //
    const absl::optional<Mesh3d>& effect_mesh_3d,  //
    ImageFrame&& effect_texture) {
  MP_RETURN_IF_ERROR(ValidateEnvironment(environment))
      << "Invalid environment!";
  if (effect_mesh_3d) {
    MP_RETURN_IF_ERROR(ValidateMesh3d(*effect_mesh_3d))
        << "Invalid effect 3D mesh!";
  }

  ASSIGN_OR_RETURN(std::unique_ptr<RenderTarget> render_target,
                   RenderTarget::Create(),
                   _ << "Failed to create a render target!");
  ASSIGN_OR_RETURN(std::unique_ptr<Renderer> renderer, Renderer::Create(),
                   _ << "Failed to create a renderer!");
  ASSIGN_OR_RETURN(RenderableMesh3d renderable_quad_mesh_3d,
                   RenderableMesh3d::CreateFromProtoMesh3d(CreateQuadMesh3d()),
                   _ << "Failed to create a renderable quad mesh!");
  absl::optional<RenderableMesh3d> renderable_effect_mesh_3d;
  if (effect_mesh_3d) {
    ASSIGN_OR_RETURN(renderable_effect_mesh_3d,
                     RenderableMesh3d::CreateFromProtoMesh3d(*effect_mesh_3d),
                     _ << "Failed to create a renderable effect mesh!");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<Texture> empty_color_gl_texture,
                   Texture::CreateFromImageFrame(CreateEmptyColorTexture()),
                   _ << "Failed to create an empty color texture!");
  ASSIGN_OR_RETURN(std::unique_ptr<Texture> effect_gl_texture,
                   Texture::CreateFromImageFrame(effect_texture),
                   _ << "Failed to create an effect texture!");

  std::unique_ptr<EffectRenderer> result =
      absl::make_unique<EffectRendererImpl>(
          environment, std::move(render_target), std::move(renderer),
          std::move(renderable_quad_mesh_3d),
          std::move(renderable_effect_mesh_3d),
          std::move(empty_color_gl_texture), std::move(effect_gl_texture));

  return result;
}

}  // namespace mediapipe::face_geometry
