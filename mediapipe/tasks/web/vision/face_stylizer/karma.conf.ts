module.exports = config => {
  config.files.push({
    pattern: 'mediapipe/tasks/**',
    watched: false,
    served: true,
    nocache: false,
    included: false,
  });
  config.pingTimeout = 400000;
};
