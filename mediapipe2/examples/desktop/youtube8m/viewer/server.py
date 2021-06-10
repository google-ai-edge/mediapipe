"""Server for YouTube8M Model Inference Demo.

Serves up both the static files for the website and provides a service that
fetches the video id and timestamp based labels for a video analyzed in a
tfrecord files.

"""
from __future__ import print_function
import json
import os
import re
import socket
import subprocess
import sys

from absl import app
from absl import flags
import http.client
import http.server
from six.moves.urllib import parse

FLAGS = flags.FLAGS
flags.DEFINE_bool("show_label_at_center", False,
                  "Show labels at the center of the segment.")
flags.DEFINE_integer("port", 8008, "Port that the API is served over.")
flags.DEFINE_string("tmp_dir", "/tmp/mediapipe",
                    "Temporary asset storage location.")
flags.DEFINE_string("root", "", "MediaPipe root directory.")
# binary, pbtxt, label_map paths are relative to 'root' path
flags.DEFINE_string(
    "binary",
    "bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference",
    "Inference binary location.")
flags.DEFINE_string(
    "pbtxt",
    "mediapipe/graphs/youtube8m/yt8m_dataset_model_inference.pbtxt",
    "Default pbtxt graph file.")
flags.DEFINE_string("label_map", "mediapipe/graphs/youtube8m/label_map.txt",
                    "Default label map text file.")


class HTTPServerV6(http.server.HTTPServer):
  address_family = socket.AF_INET6


class Youtube8MRequestHandler(http.server.SimpleHTTPRequestHandler):
  """Static file server with /healthz support."""

  def do_GET(self):
    if self.path.startswith("/healthz"):
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.send_header("Content-length", 2)
      self.end_headers()
      self.wfile.write("ok")
    if self.path.startswith("/video"):
      parsed_params = parse.urlparse(self.path)
      url_params = parse.parse_qs(parsed_params.query)

      tfrecord_path = ""
      segment_size = 5

      print(url_params)
      if "file" in url_params:
        tfrecord_path = url_params["file"][0]
      if "segments" in url_params:
        segment_size = int(url_params["segments"][0])

      self.fetch(tfrecord_path, segment_size)

    else:
      if self.path == "/":
        self.path = "/index.html"
      # Default to serve up a local file
      self.path = "/static" + self.path
      http.server.SimpleHTTPRequestHandler.do_GET(self)

  def report_error(self, msg):
    """Simplifies sending out a string as a 500 http response."""
    self.send_response(500)
    self.send_header("Content-type", "text/plain")
    self.end_headers()
    if sys.version_info[0] < 3:
      self.wfile.write(str(msg).encode("utf-8"))
    else:
      self.wfile.write(bytes(msg, "utf-8"))

  def report_missing_files(self, files):
    """Sends out 500 response with missing files."""
    accumulate = ""
    for file_path in files:
      if not os.path.exists(file_path):
        accumulate = "%s '%s'" % (accumulate, file_path)

    if accumulate:
      self.report_error("Could not find:%s" % accumulate)
      return True

    return False

  def fetch(self, path, segment_size):
    """Returns the video id and labels for a tfrecord at a provided index."""

    print("Received request.  File=", path, "Segment Size =", segment_size)

    if (self.report_missing_files([
        "%s/%s" % (FLAGS.root, FLAGS.pbtxt),
        "%s/%s" % (FLAGS.root, FLAGS.binary),
        "%s/%s" % (FLAGS.root, FLAGS.label_map)
    ])):
      return

    # Parse the youtube video id off the end of the link or as a standalone id.
    filename_match = re.match(
        "(?:.*youtube.*v=)?([a-zA-Z-0-9_]{2})([a-zA-Z-0-9_]+)", path)
    tfrecord_url = filename_match.expand(r"data.yt8m.org/2/j/r/\1/\1\2.js")

    print("Trying to get tfrecord via", tfrecord_url)

    connection = http.client.HTTPConnection("data.yt8m.org")
    connection.request("GET", tfrecord_url)
    response = connection.getresponse()

    response_object = json.loads(response.read())
    filename = response_object["filename_raw"]
    index = response_object["index"]

    print("TFRecord discovered: ", filename, ", index", index)

    output_file = r"%s/%s" % (FLAGS.tmp_dir, filename)
    tfrecord_url = r"http://us.data.yt8m.org/2/frame/train/%s" % filename

    connection = http.client.HTTPConnection("us.data.yt8m.org")
    connection.request("HEAD",
                       filename_match.expand(r"/2/frame/train/%s" % filename))
    response = connection.getresponse()
    if response.getheader("Content-Type") != "application/octet-stream":
      self.report_error("Filename '%s' is invalid." % path)

    print(output_file, "exists on yt8m.org. Did we fetch this before?")

    if not os.path.exists(output_file):
      print(output_file, "doesn't exist locally, download it now.")
      return_code = subprocess.call(
          ["curl", "--output", output_file, tfrecord_url],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)
      if return_code:
        self.report_error("Could not retrieve contents from %s" % tfrecord_url)
        return
    else:
      print(output_file, "exist locally, reuse it.")

    print("Run the graph...")
    process = subprocess.Popen([
        "%s/%s" % (FLAGS.root, FLAGS.binary),
        "--calculator_graph_config_file=%s/%s" % (FLAGS.root, FLAGS.pbtxt),
        "--input_side_packets=tfrecord_path=%s" % output_file +
        ",record_index=%d" % index + ",desired_segment_size=%d" % segment_size,
        "--output_stream=annotation_summary",
        "--output_stream_file=%s/labels" % FLAGS.tmp_dir,
        "--output_side_packets=yt8m_id",
        "--output_side_packets_file=%s/yt8m_id" % FLAGS.tmp_dir
    ],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout_str, stderr_str = process.communicate()
    process.wait()

    if stderr_str and "success" not in str(stderr_str).lower():
      self.report_error("Error executing server binary: \n%s" % stderr_str)
      return

    f = open("%s/yt8m_id" % FLAGS.tmp_dir, "r")
    contents = f.read()
    print("yt8m_id is", contents[-5:-1])

    curl_arg = "data.yt8m.org/2/j/i/%s/%s.js" % (contents[-5:-3],
                                                 contents[-5:-1])
    print("Grab labels from", curl_arg)
    process = subprocess.Popen(["curl", curl_arg],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout = process.communicate()
    process.wait()

    stdout_str = stdout[0].decode("utf-8")

    match = re.match(""".+"([^"]+)"[^"]+""", stdout_str)
    final_results = {
        "video_id": match.group(1),
        "link": "https://www.youtube.com/watch?v=%s" % match.group(1),
        "entries": []
    }
    f = open("%s/labels" % FLAGS.tmp_dir, "r")
    lines = f.readlines()
    show_at_center = FLAGS.show_label_at_center

    print("%s/labels" % FLAGS.tmp_dir, "holds", len(lines), "entries")
    for line in lines:
      entry = {"labels": []}
      final_results["entries"].append(entry)
      first = True
      for column in line.split(","):
        if first:
          subtract = segment_size / 2.0 if show_at_center else 0.0
          entry["time"] = float(int(column)) / 1000000.0 - subtract
          first = False
        else:
          label_score = re.match("(.+):([0-9.]+).*", column)
          if label_score:
            score = float(label_score.group(2))
            entry["labels"].append({
                "label": label_score.group(1),
                "score": score
            })
          else:
            print("empty score")

    response_json = json.dumps(final_results, indent=2, separators=(",", ": "))
    self.send_response(200)
    self.send_header("Content-type", "application/json")
    self.end_headers()
    if sys.version_info[0] < 3:
      self.wfile.write(str(response_json).encode("utf-8"))
    else:
      self.wfile.write(bytes(response_json, "utf-8"))


def update_pbtxt():
  """Update graph.pbtxt to use full path to label_map.txt."""
  edited_line = ""
  lines = []
  with open("%s/%s" % (FLAGS.root, FLAGS.pbtxt), "r") as f:
    lines = f.readlines()
    for line in lines:
      if "label_map_path" in line:
        kv = line.split(":")
        edited_line = kv[0] + (": \"%s/%s\"\n" % (FLAGS.root, FLAGS.label_map))
  with open("%s/%s" % (FLAGS.root, FLAGS.pbtxt), "w") as f:
    for line in lines:
      if "label_map_path" in line:
        f.write(edited_line)
      else:
        f.write(line)


def main(unused_args):
  dname = os.path.dirname(os.path.abspath(__file__))
  os.chdir(dname)
  if not FLAGS.root:
    print("Must specify MediaPipe root directory: --root `pwd`")
    return
  update_pbtxt()
  port = FLAGS.port
  print("Listening on port %s" % port)  # pylint: disable=superfluous-parens
  server = HTTPServerV6(("::", int(port)), Youtube8MRequestHandler)
  server.serve_forever()


if __name__ == "__main__":
  app.run(main)
