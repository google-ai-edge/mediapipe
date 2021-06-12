/**
 * @license
 * Copyright 2019 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const STATE_PLAYER=0;
const STATE_COVER=1;
const STATE_SPINNER=2;

/**
* Looks up the value of a url parameter.
*
* @param {string} param The name of the parameter.
* @return {?string} The parameter value or null if there is no such parameter.
*/
var getUrlParameter = function(param) {
    const url = decodeURIComponent(window.location.search.substring(1));
    const url_parts = url.split('&');
    for (var i = 0; i < url_parts.length; i++) {
        const param_name = url_parts[i].split(/=(.*)/);
        if (param_name[0] === param) {
            return param_name[1] === undefined ? null : param_name[1];
        }
    }
};

/**
* Sets the fields in the form to match the values of the URL parameters.
*/
const updateFormFromURL = function() {
  const form_elements = document.getElementById('form').elements;
  const url = decodeURIComponent(window.location.search.substring(1));
  const url_parts = url.split('&');
  for (var i = 0; i < url_parts.length; i++) {
    const p = url_parts[i].split(/=(.*)/);
    if (p.length >= 2) {
      if (form_elements[p[0]]) {
        form_elements[p[0]].value = decodeURIComponent(p[1]);
      }
    }
  }
};

let player = null;
let intervalID = undefined;
let entries = [];

/**
 * Constructs the embedded YouTube player.
 */
window.onYouTubeIframeAPIReady = () => {
  player = new YT.Player('ytplayer', {
    events: {
      'onReady': onPlayerReady,
      'onStateChange': onStateChange
    }
  });
};


/**
 * Listens for YouTube video events. When video is playing, periodically checks
 * the time signature and updates the feedback with labels. When video stops,
 * shuts off interval timer to save cycles.
 * @param {!Event} event YouTube API Event.
 */
function onStateChange(event) {
  if (event.data === 1) {
    // Youtube switched to playing.
    intervalID = setInterval(function(){
      const currentTime = player.getCurrentTime();
      let winner = undefined;
      let first = undefined;
      for (entry of entries) {
        if (!first) {
          first = entry.labels;
        }
        if (entry.time < currentTime) {
          winner = entry.labels;
        } else {
          break;
        }
      }
      if (!winner) {
        winner = first;
      }
      const threshold =
          document.getElementById('form').elements['threshold'].value;
      let message = "";
      for (var label of winner) {
        if (label.score >= threshold) {
          message =  `${message}${label.label} (score: ${label.score})\n`;
        }
      }
      $("textarea#feedback").val(message);
    });
  } else {
    if (intervalID) {
      clearInterval(intervalID);
    }
  }
}

/**
 * Turns elements of the player on and off to reflect the state of the "app".
 * @param {number} state One of STATE_COVER | STATE_SPINNER | STATE_PLAYER.
 */
function showState(state) {
  switch(state) {
    case STATE_COVER:
      $('#cover').show();
      $('#spinner').hide();
      $('#ytplayer').hide();
      break;
    case STATE_SPINNER:
      $('#cover').hide();
      $('#spinner').show();
      $('#ytplayer').hide();
      break;
    case STATE_PLAYER:
    default:
      $('#cover').hide();
      $('#spinner').hide();
      $('#ytplayer').show();
      break;
  }
}

/**
 * Hide error field and clear its message.
 */
function hideError() {
  $('#error_msg').css("visibility", "hidden").text('');
}

/**
 * Set the error to visible and set its message.
 * @param {string} msg Error message as a string.
 */
function showError(msg) {
  $('#error_msg').css("visibility", "visible").text(msg);
}

/**
 * Privides numeric feedback for the slider.
 */
function connectSlider() {
  $('#threshold_label').text(
      `Score Threshold (${$('#threshold')[0].value})`);
  $('#threshold').on('input', () => {
    $('#threshold_label').text(
        `Score Threshold (${$('#threshold')[0].value})`);
  });
  $('#segments_label').text(
      `Segment Size (${$('#segments')[0].value})`);
  $('#segments').on('input', () => {
    $('#segments_label').text(
        `Segment Size (${$('#segments')[0].value})`);
  });
}

/**
 * Retrieve video information from backend.
 * @param {string} filePath name of a tfrecord file.
 * @param {number} segments desired number of segments (1-300)
 */
function fetchVideo(filePath, segments) {
  const url = "/video?file=" + filePath + "&segments=" + segments;
  $.ajax({
    url: url,
    success: function(result) {
      const videoId = result["video_id"];
      player.loadVideoById(videoId);
      entries = result['entries'];
      showState(STATE_PLAYER);
    },
    error: (err) => {
      showState(STATE_COVER);
      console.log(err);
      showError(err.responseText);
    },
    datatype: "json"
  });
}

/**
 * Called when the embedded YouTube player has finished loading. It loads the
 * requested video into the player and calls the golden6_viewer API to retrieve
 * the frame-level data for that video.
 */
function onPlayerReady() {
  const filePath = getUrlParameter('file') || "";
  const segments = parseInt(getUrlParameter('segments')) || 0;

  updateFormFromURL();
  hideError();
  connectSlider();

  if (!filePath) {
    return;
  }

  showState(STATE_SPINNER);
  fetchVideo(filePath, segments);
}
