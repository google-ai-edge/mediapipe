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

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Class for parsing a single .obj file into openGL-usable pieces.
 *
 * <p>Usage:
 *
 * <p>SimpleObjParser objParser = new SimpleObjParser("animations/cow/cow320.obj", .015f);
 *
 * <p>if (objParser.parse()) { ... }
 */
public class SimpleObjParser {
  private static class ShortPair {
    private final Short first;
    private final Short second;

    public ShortPair(Short newFirst, Short newSecond) {
      first = newFirst;
      second = newSecond;
    }

    public Short getFirst() {
      return first;
    }

    public Short getSecond() {
      return second;
    }
  }

  private static final String TAG = SimpleObjParser.class.getSimpleName();
  private static final boolean DEBUG = false;
  private static final int INVALID_INDEX = -1;
  private static final int POSITIONS_COORDS_PER_VERTEX = 3;
  private static final int TEXTURE_COORDS_PER_VERTEX = 2;
  private final String fileName;

  // Since .obj doesn't tie together texture coordinates and vertex
  // coordinates, but OpenGL does, we need to keep a map of all such pairings that occur in
  // our face list.
  private final HashMap<ShortPair, Short> vertexTexCoordMap;

  // Internal (de-coupled) unique vertices and texture coordinates
  private ArrayList<Float> vertices;
  private ArrayList<Float> textureCoords;

  // Data we expose to openGL for rendering
  private float[] finalizedVertices;
  private float[] finalizedTextureCoords;
  private ArrayList<Short> finalizedTriangles;

  // So we only display warnings about dropped w-coordinates once
  private boolean vertexCoordIgnoredWarning;
  private boolean textureCoordIgnoredWarning;
  private boolean startedProcessingFaces;

  private int numPrimitiveVertices;
  private int numPrimitiveTextureCoords;
  private int numPrimitiveFaces;

  // For scratchwork, so we don't have to keep reallocating
  private float[] tempCoords;

  // We scale all our position coordinates uniformly by this factor
  private float objectUniformScaleFactor;

  public SimpleObjParser(String objFile, float scaleFactor) {
    objectUniformScaleFactor = scaleFactor;

    fileName = objFile;
    vertices = new ArrayList<Float>();
    textureCoords = new ArrayList<Float>();

    vertexTexCoordMap = new HashMap<ShortPair, Short>();
    finalizedTriangles = new ArrayList<Short>();

    tempCoords = new float[Math.max(POSITIONS_COORDS_PER_VERTEX, TEXTURE_COORDS_PER_VERTEX)];
    numPrimitiveFaces = 0;

    vertexCoordIgnoredWarning = false;
    textureCoordIgnoredWarning = false;
    startedProcessingFaces = false;
  }

  // Simple helper wrapper function
  private void debugLogString(String message) {
    if (DEBUG) {
      System.out.println(message);
    }
  }

  private void parseVertex(String[] linePieces) {
    // Note: Traditionally xyzw is acceptable as a format, with w defaulting to 1.0, but for now
    // we only parse xyz.
    if (linePieces.length < POSITIONS_COORDS_PER_VERTEX + 1
        || linePieces.length > POSITIONS_COORDS_PER_VERTEX + 2) {
      System.out.println("Malformed vertex coordinate specification, assuming xyz format only.");
      return;
    } else if (linePieces.length == POSITIONS_COORDS_PER_VERTEX + 2 && !vertexCoordIgnoredWarning) {
      System.out.println(
          "Only x, y, and z parsed for vertex coordinates; w coordinates will be ignored.");
      vertexCoordIgnoredWarning = true;
    }

    boolean success = true;
    try {
      for (int i = 1; i < POSITIONS_COORDS_PER_VERTEX + 1; i++) {
        tempCoords[i - 1] = Float.parseFloat(linePieces[i]);
      }
    } catch (NumberFormatException e) {
      success = false;
      System.out.println("Malformed vertex coordinate error: " + e.toString());
    }

    if (success) {
      for (int i = 0; i < POSITIONS_COORDS_PER_VERTEX; i++) {
        vertices.add(Float.valueOf(tempCoords[i] * objectUniformScaleFactor));
      }
    }
  }

  private void parseTextureCoordinate(String[] linePieces) {
    // Similar to vertices, uvw is acceptable as a format, with w defaulting to 0.0, but for now we
    // only parse uv.
    if (linePieces.length < TEXTURE_COORDS_PER_VERTEX + 1
        || linePieces.length > TEXTURE_COORDS_PER_VERTEX + 2) {
      System.out.println("Malformed texture coordinate specification, assuming uv format only.");
      return;
    } else if (linePieces.length == (TEXTURE_COORDS_PER_VERTEX + 2)
        && !textureCoordIgnoredWarning) {
      debugLogString("Only u and v parsed for texture coordinates; w coordinates will be ignored.");
      textureCoordIgnoredWarning = true;
    }

    boolean success = true;
    try {
      for (int i = 1; i < TEXTURE_COORDS_PER_VERTEX + 1; i++) {
        tempCoords[i - 1] = Float.parseFloat(linePieces[i]);
      }
    } catch (NumberFormatException e) {
      success = false;
      System.out.println("Malformed texture coordinate error: " + e.toString());
    }

    if (success) {
      // .obj files treat (0,0) as top-left, compared to bottom-left for openGL.  So invert "v"
      // texture coordinate only here.
      textureCoords.add(Float.valueOf(tempCoords[0]));
      textureCoords.add(Float.valueOf(1.0f - tempCoords[1]));
    }
  }

  // Will return INVALID_INDEX if error occurs, and otherwise will return finalized (combined)
  // index, adding and hashing new combinations as it sees them.
  private short parseAndProcessCombinedVertexCoord(String coordString) {
    String[] coords = coordString.split("/");
    try {
      // Parse vertex and texture indices; 1-indexed from front if positive and from end of list if
      // negative.
      short vertexIndex = Short.parseShort(coords[0]);
      short textureIndex = Short.parseShort(coords[1]);
      if (vertexIndex > 0) {
        vertexIndex--;
      } else {
        vertexIndex = (short) (vertexIndex + numPrimitiveVertices);
      }
      if (textureIndex > 0) {
        textureIndex--;
      } else {
        textureIndex = (short) (textureIndex + numPrimitiveTextureCoords);
      }

      // Combine indices and look up in pair map.
      ShortPair indexPair = new ShortPair(Short.valueOf(vertexIndex), Short.valueOf(textureIndex));
      Short combinedIndex = vertexTexCoordMap.get(indexPair);
      if (combinedIndex == null) {
        short numIndexPairs = (short) vertexTexCoordMap.size();
        vertexTexCoordMap.put(indexPair, numIndexPairs);
        return numIndexPairs;
      } else {
        return combinedIndex.shortValue();
      }
    } catch (NumberFormatException e) {
      // Failure to parse coordinates as shorts
      return INVALID_INDEX;
    }
  }

  // Note: it is assumed that face list occurs AFTER vertex and texture coordinate lists finish in
  //  the obj file format.
  private void parseFace(String[] linePieces) {
    if (linePieces.length < 4) {
      System.out.println("Malformed face index list: there must be at least 3 indices per face");
      return;
    }

    short[] faceIndices = new short[linePieces.length - 1];
    boolean success = true;
    for (int i = 1; i < linePieces.length; i++) {
      short faceIndex = parseAndProcessCombinedVertexCoord(linePieces[i]);

      if (faceIndex < 0) {
        System.out.println(faceIndex);
        System.out.println("Malformed face index: " + linePieces[i]);
        success = false;
        break;
      }
      faceIndices[i - 1] = faceIndex;
    }

    if (success) {
      numPrimitiveFaces++;
      // Manually triangulate the face under the assumption that the points are coplanar, the poly
      // is convex, and the points are listed in either clockwise or anti-clockwise orientation.
      for (int i = 1; i < faceIndices.length - 1; i++) {
        // We use a triangle fan here, so first point is part of all triangles
        finalizedTriangles.add(faceIndices[0]);
        finalizedTriangles.add(faceIndices[i]);
        finalizedTriangles.add(faceIndices[i + 1]);
      }
    }
  }

  // Iterate over map and reconstruct proper vertex/texture coordinate pairings.
  private boolean constructFinalCoordinatesFromMap() {
    final int numIndexPairs = vertexTexCoordMap.size();
    // XYZ vertices and UV texture coordinates
    finalizedVertices = new float[POSITIONS_COORDS_PER_VERTEX * numIndexPairs];
    finalizedTextureCoords = new float[TEXTURE_COORDS_PER_VERTEX * numIndexPairs];
    try {
      for (Map.Entry<ShortPair, Short> entry : vertexTexCoordMap.entrySet()) {
        ShortPair indexPair = entry.getKey();
        short rawVertexIndex = indexPair.getFirst().shortValue();
        short rawTexCoordIndex = indexPair.getSecond().shortValue();
        short finalIndex = entry.getValue().shortValue();
        for (int i = 0; i < POSITIONS_COORDS_PER_VERTEX; i++) {
          finalizedVertices[POSITIONS_COORDS_PER_VERTEX * finalIndex + i]
              = vertices.get(rawVertexIndex * POSITIONS_COORDS_PER_VERTEX + i);
        }
        for (int i = 0; i < TEXTURE_COORDS_PER_VERTEX; i++) {
          finalizedTextureCoords[TEXTURE_COORDS_PER_VERTEX * finalIndex + i]
              = textureCoords.get(rawTexCoordIndex * TEXTURE_COORDS_PER_VERTEX + i);
        }
      }
    } catch (NumberFormatException e) {
      System.out.println("Malformed index in vertex/texture coordinate mapping.");
      return false;
    }
    return true;
  }

  /**
   * Returns the vertex position coordinate list (x1, y1, z1, x2, y2, z2, ...) after a successful
   * call to parse().
   */
  public float[] getVertices() {
    return finalizedVertices;
  }

  /**
   * Returns the vertex texture coordinate list (u1, v1, u2, v2, ...) after a successful call to
   * parse().
   */
  public float[] getTextureCoords() {
    return finalizedTextureCoords;
  }

  /**
   * Returns the list of indices (a1, b1, c1, a2, b2, c2, ...) after a successful call to parse().
   * Each (a, b, c) triplet specifies a triangle to be rendered, with a, b, and c Short objects used
   * to index into the coordinates returned by getVertices() and getTextureCoords().<p></p>
   * For example, a Short index representing 5 should be used to index into vertices[15],
   * vertices[16], and vertices[17], as well as textureCoords[10] and textureCoords[11].
   */
  public ArrayList<Short> getTriangles() {
    return finalizedTriangles;
  }

  /**
   * Attempts to locate and read the specified .obj file, and parse it accordingly.  None of the
   * getter functions in this class will return valid results until a value of true is returned
   * from this function.
   * @return true on success.
   */
  public boolean parse() {
    boolean success = true;
    BufferedReader reader = null;
    try {
      reader = Files.newBufferedReader(Paths.get(fileName), UTF_8);
      String line;
      while ((line = reader.readLine()) != null) {
        // Skip over lines with no characters
        if (line.length() < 1) {
          continue;
        }

        // Ignore comment lines entirely
        if (line.charAt(0) == '#') {
          continue;
        }

        // Split into pieces based on whitespace, and process according to first command piece
        String[] linePieces = line.split(" +");
        switch (linePieces[0]) {
          case "v":
            // Add vertex
            if (startedProcessingFaces) {
              throw new IOException("Vertices must all be declared before faces in obj files.");
            }
            parseVertex(linePieces);
            break;
          case "vt":
            // Add texture coordinate
            if (startedProcessingFaces) {
              throw new IOException(
                  "Texture coordinates must all be declared before faces in obj files.");
            }
            parseTextureCoordinate(linePieces);
            break;
          case "f":
            // Vertex and texture coordinate lists should be locked into place by now
            if (!startedProcessingFaces) {
              startedProcessingFaces = true;
              numPrimitiveVertices = vertices.size() / POSITIONS_COORDS_PER_VERTEX;
              numPrimitiveTextureCoords = textureCoords.size() / TEXTURE_COORDS_PER_VERTEX;
            }
            // Add face
            parseFace(linePieces);
            break;
          default:
            // Unknown or unused directive: ignoring
            // Note: We do not yet process vertex normals or curves, so we ignore {vp, vn, s}
            // Note: We assume only a single object, so we ignore {g, o}
            // Note: We also assume a single texture, which we process independently, so we ignore
            // {mtllib, usemtl}
            break;
        }
      }

      // If we made it all the way through, then we have a vertex-to-tex-coord pair mapping, so
      // construct our final vertex and texture coordinate lists now.
      success = constructFinalCoordinatesFromMap();

    } catch (IOException e) {
      success = false;
      System.out.println("Failure to parse obj file: " + e.toString());
    } finally {
      try {
        if (reader != null) {
          reader.close();
        }
      } catch (IOException e) {
        System.out.println("Couldn't close reader");
      }
    }
    if (success) {
      debugLogString("Successfully parsed " + numPrimitiveVertices + " vertices and "
          + numPrimitiveTextureCoords + " texture coordinates into " + vertexTexCoordMap.size()
          + " combined vertices and " + numPrimitiveFaces + " faces, represented as a mesh of "
          + finalizedTriangles.size() / 3 + " triangles.");
    }
    return success;
  }
}
