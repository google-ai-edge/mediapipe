// Copyright 2021 The MediaPipe Authors.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class for running desktop-side parsing/packing routines on .obj AR assets. Usage: ObjParser
 * --input_dir=[INPUT_DIRECTORY] --output_dir=[OUTPUT_DIRECTORY] where INPUT_DIRECTORY is the folder
 * with asset obj files to process, and OUTPUT_DIRECTORY is the folder where processed asset uuu
 * file should be placed.
 *
 * <p>NOTE: Directories are assumed to be absolute paths.
 */
public final class ObjParserMain {
  // Simple FileFilter implementation to let us walk over only our .obj files in a particular
  // directory.
  private static final class ObjFileFilter implements FileFilter {
    ObjFileFilter() {
      // Nothing to do here.
    }

    @Override
    public boolean accept(File file) {
      return file.getName().endsWith(".obj");
    }
  }

  // File extension for binary output files; tagged onto end of initial file extension.
  private static final String BINARY_FILE_EXT = ".uuu";
  private static final String INPUT_DIR_FLAG = "--input_dir=";
  private static final String OUTPUT_DIR_FLAG = "--output_dir=";
  private static final float DEFAULT_VERTEX_SCALE_FACTOR = 30.0f;
  private static final double NS_TO_SECONDS = 1e9;

  public final PrintWriter writer;

  public ObjParserMain() {
    super();
    this.writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out, UTF_8)));
  }

  // Simple overridable logging function.
  protected void logString(String infoLog) {
    writer.println(infoLog);
  }

  /*
   * Main program logic: parse command-line arguments and perform actions.
   */
  public void run(String inDirectory, String outDirectory) {
    if (inDirectory.isEmpty()) {
      logString("Error: Must provide input directory with " + INPUT_DIR_FLAG);
      return;
    }
    if (outDirectory.isEmpty()) {
      logString("Error: Must provide output directory with " + OUTPUT_DIR_FLAG);
      return;
    }

    File dirAsFile = new File(inDirectory);
    ObjFileFilter objFileFilter = new ObjFileFilter();
    File[] objFiles = dirAsFile.listFiles(objFileFilter);

    FileOutputStream outputStream = null;
    logString("Parsing directory: " + inDirectory);
    // We need frames processed in correct order.
    Arrays.sort(objFiles);

    for (File objFile : objFiles) {
      String fileName = objFile.getAbsolutePath();

      // Just take the file name of the first processed frame.
      if (outputStream == null) {
        String outputFileName = outDirectory + objFile.getName() + BINARY_FILE_EXT;
        try {
          // Create new file here, if we can.
          outputStream = new FileOutputStream(outputFileName);
          logString("Created outfile: " + outputFileName);
        } catch (Exception e) {
          logString("Error creating outfile: " + e.toString());
          e.printStackTrace(writer);
          return;
        }
      }

      // Process each file into the stream.
      logString("Processing file: " + fileName);
      processFile(fileName, outputStream);
    }

    // Finally close the stream out.
    try {
      if (outputStream != null) {
        outputStream.close();
      }
    } catch (Exception e) {
      logString("Error trying to close output stream: " + e.toString());
      e.printStackTrace(writer);
    }
  }

  /*
   * Entrypoint for command-line executable.
   */
  public static void main(String[] args) {
    // Parse flags
    String inDirectory = "";
    String outDirectory = "";
    for (int i = 0; i < args.length; i++) {
      if (args[i].startsWith(INPUT_DIR_FLAG)) {
        inDirectory = args[i].substring(INPUT_DIR_FLAG.length());
        // Make sure this will be treated as a directory
        if (!inDirectory.endsWith("/")) {
          inDirectory += "/";
        }
      }
      if (args[i].startsWith(OUTPUT_DIR_FLAG)) {
        outDirectory = args[i].substring(OUTPUT_DIR_FLAG.length());
        // Make sure this will be treated as a directory
        if (!outDirectory.endsWith("/")) {
          outDirectory += "/";
        }
      }
    }
    ObjParserMain parser = new ObjParserMain();
    parser.run(inDirectory, outDirectory);
    parser.writer.flush();
  }

  /*
   * Internal helper function to parse a .obj from an infile name and stream the resulting data
   * directly out in binary-dump format to outputStream.
   */
  private void processFile(String infileName, OutputStream outputStream) {
    long start = System.nanoTime();

    // First we parse the obj.
    SimpleObjParser objParser = new SimpleObjParser(infileName, DEFAULT_VERTEX_SCALE_FACTOR);
    if (!objParser.parse()) {
      logString("Error parsing .obj file before processing");
      return;
    }

    final float[] vertices = objParser.getVertices();
    final float[] textureCoords = objParser.getTextureCoords();
    final ArrayList<Short> triangleList = objParser.getTriangles();

    // Overall byte count to stream: 12 for the 3 list-length ints, and then 4 for each vertex and
    // texCoord int, and finally 2 for each triangle index short.
    final int bbSize =
        12 + 4 * vertices.length + 4 * textureCoords.length + 2 * triangleList.size();

    // Ensure ByteBuffer is native order, just like we want to read it in, but is NOT direct, so
    // we can call .array() on it.
    ByteBuffer bb = ByteBuffer.allocate(bbSize);
    bb.order(ByteOrder.nativeOrder());

    bb.putInt(vertices.length);
    bb.putInt(textureCoords.length);
    bb.putInt(triangleList.size());
    logString(String.format("Writing... Vertices: %d, TextureCoords: %d, Indices: %d.%n",
        vertices.length, textureCoords.length, triangleList.size()));
    for (float vertex : vertices) {
      bb.putFloat(vertex);
    }
    for (float textureCoord : textureCoords) {
      bb.putFloat(textureCoord);
    }
    for (Short vertexIndex : triangleList) {
      bb.putShort(vertexIndex.shortValue());
    }
    bb.position(0);
    try {
      outputStream.write(bb.array(), 0, bbSize);
      logString(String.format("Processing successful!  Took %.4f seconds.%n",
          (System.nanoTime() - start) / NS_TO_SECONDS));
    } catch (Exception e) {
      logString("Error writing during processing: " + e.toString());
      e.printStackTrace(writer);
    }
  }
}
