## Visualizing MediaPipe Graphs

-   [Working within the editor](#working-within-the-editor)
-   [Understanding the Graph](#understanding-the-graph)

To help users understand the structure of their calculator graphs and to
understand the overall behavior of their machine learning inference pipelines,
we have built the [MediaPipe Visualizer](https://mediapipe-viz.appspot.com/) that is available online.

*   A graph view allows users to see a connected calculator graph as expressed
    through a graph configuration that is pasted into the graph editor or
    uploaded. The user can visualize and troubleshoot a graph they have created.

    ![Startup screen](./images/startup_screen.png){width="800"}

### Working within the editor

Getting Started:

The graph can be modified by adding and editing code in the Editor view.

![Editor UI](./images/editor_view.png){width="600"}

*   Pressing the "New" button in the upper right corner will clear any existing
    code in the Editor window.

    ![New Button](./images/upload_button.png){width="300"}

*   Pressing the "Upload" button will prompt the user to select a local PBTXT
    file, which will everwrite the current code within the editor.

*   Alternatively, code can be pasted directly into the editor window.

*   Errors and informational messages will appear in the Feedback window.

    ![Error Msg](./images/console_error.png){width="400"}

### Understanding the Graph

The visualizer graph shows the connections between calculator nodes.

*   Streams exit from the bottom of the calculator producing the stream and
    enter the top of any calculator receiving the stream. (Notice the use of the
    key, "input_stream" and "output_stream").

    ![Stream UI](./images/stream_ui.png){width="350"}
    ![Stream_code](./images/stream_code.png){width="350"}

*   Sidepackets work the same, except that they exit a node on the right and
    enter on the left. (Notice the use of the key, "input_side_packet" and
    "output_side_packet").

    ![Sidepacket UI](./images/side_packet.png){width="350"}
    ![Sidepacket_code](./images/side_packet_code.png){width="350"}

*   There are special nodes that represent inputs and outputs to the graph and
    can supply either side packets or streams.

    ![Special nodes](./images/special_nodes.png){width="350"}
    ![Special nodes](./images/special_nodes_code.png){width="350"}

