import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';

export default {
  plugins: [
    resolve(),
    commonjs(),
    terser()
  ]
}
