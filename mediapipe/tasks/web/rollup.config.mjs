import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import replace from '@rollup/plugin-replace';
import terser from '@rollup/plugin-terser';

export default {
  plugins: [
    // Workaround for https://github.com/protocolbuffers/protobuf-javascript/issues/151
    replace({
      'var calculator_options_pb = {};': 'var calculator_options_pb = {}; var mediapipe_framework_calculator_options_pb = calculator_options_pb;',
      delimiters: ['', '']
    }),
    resolve(),
    commonjs(),
    terser()
  ]
}
