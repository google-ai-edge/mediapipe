import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import replace from '@rollup/plugin-replace';

export default {
  output: {
    name: 'bundle',
    sourcemap: false
  },
  plugins: [
    // Workaround for https://github.com/protocolbuffers/protobuf-javascript/issues/151
    replace({
      'var calculator_options_pb = {};': 'var calculator_options_pb = {}; var mediapipe_framework_calculator_options_pb = calculator_options_pb;',
      delimiters: ['', '']
    }),
    resolve({browser: true}),
    commonjs(),
    terser()
  ]
}
