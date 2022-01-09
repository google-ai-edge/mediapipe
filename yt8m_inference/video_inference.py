import tensorflow as tf
import numpy as np 
import os


def read_pb_file(pb_file_path):
    sequence_example = open(pb_file_path, 'rb').read()
    sequence = tf.train.SequenceExample.FromString(sequence_example)
    return sequence

def parse_pb_sequence(pb_file_path):
    seq = read_pb_file(pb_file_path)

    rgb_features = seq.feature_lists.feature_list['RGB/feature/floats'].feature
    #print(len(rgb_features), len(rgb_features[0].float_list.value))
    rgb_features_array = np.array([rgb_feature.float_list.value for rgb_feature in rgb_features])

    audio_features = seq.feature_lists.feature_list['AUDIO/feature/floats'].feature
    #print(len(audio_features), len(audio_features[0].float_list.value))
    audio_features_array = np.array([audio_feature.float_list.value for audio_feature in audio_features])
    
    n = min(rgb_features_array.shape[0], audio_features_array.shape[0])

    rgb_features_array, audio_features_array = rgb_features_array[:n, :], audio_features_array[:n, :]

    concatenated_features = np.concatenate([rgb_features_array, audio_features_array], axis=1)

    return concatenated_features

def individual_frames_to_segments(frames_feature_matrix, segment_size=5):
    num_frames, descriptor_dimensionality = frames_feature_matrix.shape
    num_segments = int(num_frames / segment_size)
    num_frames_included =  num_segments*segment_size
    frames_feature_matrix = frames_feature_matrix[:num_frames_included, :]
    segmented_version = frames_feature_matrix.reshape([num_segments, segment_size, descriptor_dimensionality])
    return segmented_version

class VideoInference:
    def __init__(self) -> None:
        model_path = "/tmp/mediapipe/saved_model"
        self.sess = tf.Session()
        meta_graph = tf.saved_model.load(export_dir=model_path, sess=self.sess, tags=['serve'])
        sig_def = meta_graph.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        self.num_frames_placeholder = sig_def.inputs['num_frames'].name
        self.rgb_and_audio_placeholder = sig_def.inputs['rgb_and_audio'].name
        self.output_name = sig_def.outputs['predictions'].name
        self.labels = self.get_label_map()

    def get_label_map(self):
        with open("/mediapipe/mediapipe/graphs/youtube8m/label_map.txt", 'rb') as file:
            lines = file.readlines()
            labels = [line.rstrip().decode("utf-8") for line in lines]
            return labels

    def extract_video_features(self, video_path):
        command = "/mediapipe/extract_video_features.sh " +  video_path
        print(command)
        os.system(command)
        

    def infer(self, features_pb_filepath):
        f = parse_pb_sequence(features_pb_filepath)
        #print(f.shape)
        rgb_and_audio_segments = individual_frames_to_segments(f, segment_size=6)
        #print(rgb_and_audio_segments.shape)
        num_frames_array = np.ones(shape=[rgb_and_audio_segments.shape[0], 1], dtype=np.int32) * rgb_and_audio_segments.shape[1]
        with tf.Graph().as_default() as g:
            predictions = self.sess.run(self.output_name, feed_dict= {self.num_frames_placeholder: num_frames_array, self.rgb_and_audio_placeholder:rgb_and_audio_segments})
        return predictions

    def aggregate_scores(self, predictions):
        top_prediction_per_frame = predictions.argmax(axis=-1)
        top_scores = predictions.max(axis=-1)
        top_prediction_per_frame[np.where(top_scores < 0.75)] = -1

        u, count = np.unique(top_prediction_per_frame, return_counts=True)
        count_sort_ind = np.argsort(-count)
        sorted_counts = count[count_sort_ind]
        sorted_predictions = u[count_sort_ind]
        print(sorted_predictions[:3], sorted_counts[:3])
        top_prediction = sorted_predictions[0]
        if top_prediction == -1:
            top_prediction = sorted_predictions[1]
        label = self.labels[top_prediction]

        return label

    def extract_features_and_infer(self, video_path):
        self.extract_video_features(video_path)
        print("Features Extracted, about to run inference")
        predictions = self.infer('/tmp/mediapipe/features.pb')
        video_label = v.aggregate_scores(predictions)
        return video_label





if __name__ == "__main__":
    import glob
    import os
    v = VideoInference()
    videos = glob.glob("/shared_volume/test_videos/*mp4")
    print(f"test_videos: {videos}")
    d = {}
    for video in videos:
        video_label = v.extract_features_and_infer(video)
        print(video_label)
        d[os.path.basename(video)] = video_label
    
    print(f"results: {d}")