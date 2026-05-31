using System;
using System.IO;
using UnityEngine;

namespace NBodyVisualizer
{
    /// <summary>
    /// Streaming reader for the nbody-sim <b>frames.bin</b> binary format.
    ///
    /// File layout:
    ///   Header (32 bytes):
    ///     magic       char[4]   "NBOD"
    ///     version     uint32    1 or 2
    ///     n_particles uint32
    ///     n_frames    uint32
    ///     dt          double    simulation timestep (T0)
    ///     output_dt   double    time between frames (T0)
    ///
    ///   Per frame (v1):
    ///     time_T0     double    simulation time of this frame
    ///     x[n]        float[n]  x positions (AU)
    ///     y[n]        float[n]  y positions (AU)
    ///     z[n]        float[n]  z positions (AU)
    ///
    ///   Per frame (v2): adds per-particle aggregate-size field for visualization
    ///     time_T0     double
    ///     x[n]        float[n]
    ///     y[n]        float[n]
    ///     z[n]        float[n]
    ///     agg_size[n] uint16[n] (1 = singleton, N = member of an N-particle clump)
    ///
    /// The disk lies in the XZ plane; Y is the vertical (thickness) axis.
    /// This maps directly to Unity's coordinate system (Y-up).
    /// </summary>
    public sealed class NBodyReader : IDisposable
    {
        // ----------------------------------------------------------------
        // Public types
        // ----------------------------------------------------------------

        public struct Header
        {
            public uint   Version;
            public int    NParticles;
            public int    NFrames;
            public double Dt;       // simulation dt (T0)
            public double OutputDt; // time between written frames (T0)
        }

        // ----------------------------------------------------------------
        // Properties
        // ----------------------------------------------------------------

        public Header FileHeader     { get; private set; }
        public long   FrameSizeBytes { get; private set; }

        /// <summary>True if the file includes per-particle aggregate-size info (v2+).</summary>
        public bool HasAggregateSizes { get; private set; }

        // ----------------------------------------------------------------
        // Private
        // ----------------------------------------------------------------

        private const int HeaderBytes = 32;

        private readonly FileStream   _fs;
        private readonly BinaryReader _br;

        private byte[] _readBuf;

        // Three independent component buffers (xs/ys/zs MUST be separate — see
        // history: an earlier bug shared one buffer and collapsed all particles
        // onto the diagonal X=Y=Z line).
        private float[]  _xs;
        private float[]  _ys;
        private float[]  _zs;
        private ushort[] _aggBuf;   // uint16 aggregate sizes (v2 only)

        // ----------------------------------------------------------------
        // Construction
        // ----------------------------------------------------------------

        public NBodyReader(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"frames.bin not found: {path}");

            _fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                 bufferSize: 1 << 20); // 1 MB I/O buffer
            _br = new BinaryReader(_fs);

            // --- Parse header ---
            var magic = new string(_br.ReadChars(4));
            if (magic != "NBOD")
                throw new InvalidDataException(
                    $"Invalid file (expected magic 'NBOD', got '{magic}'). " +
                    $"Is this a frames.bin produced by nbody-sim?");

            uint version = _br.ReadUInt32();
            if (version != 1u && version != 2u)
                throw new InvalidDataException(
                    $"Unsupported frames.bin version {version}. Supported: 1, 2.");

            FileHeader = new Header
            {
                Version    = version,
                NParticles = (int)_br.ReadUInt32(),
                NFrames    = (int)_br.ReadUInt32(),
                Dt         = _br.ReadDouble(),
                OutputDt   = _br.ReadDouble()
            };

            HasAggregateSizes = (version >= 2u);

            // v1: time(8) + 3 * n * 4
            // v2: + n * 2
            long bytesPerFrame = 8L + (long)FileHeader.NParticles * 3 * sizeof(float);
            if (HasAggregateSizes)
                bytesPerFrame += (long)FileHeader.NParticles * sizeof(ushort);
            FrameSizeBytes = bytesPerFrame;

            _readBuf = new byte[FileHeader.NParticles * sizeof(float)];
        }

        // ----------------------------------------------------------------
        // Reading
        // ----------------------------------------------------------------

        /// <summary>
        /// Reads positions for frame <paramref name="frameIndex"/> into
        /// <paramref name="positions"/>. If <paramref name="aggSizes"/> is non-null
        /// AND the file is v2+, per-particle aggregate sizes are also read into it.
        /// For v1 files, aggSizes (if provided) is filled with 1s (all singletons).
        /// Returns the simulation time of the frame (T0).
        /// </summary>
        public double ReadFrame(int frameIndex, Vector3[] positions, ushort[] aggSizes = null)
        {
            if (frameIndex < 0 || frameIndex >= FileHeader.NFrames)
                throw new ArgumentOutOfRangeException(nameof(frameIndex),
                    $"Frame {frameIndex} out of range [0, {FileHeader.NFrames - 1}].");
            if (positions == null || positions.Length < FileHeader.NParticles)
                throw new ArgumentException("positions array is null or too small.");
            if (aggSizes != null && aggSizes.Length < FileHeader.NParticles)
                throw new ArgumentException("aggSizes array is too small.");

            long offset = HeaderBytes + (long)frameIndex * FrameSizeBytes;
            _fs.Seek(offset, SeekOrigin.Begin);

            double time = _br.ReadDouble();
            int    n    = FileHeader.NParticles;

            if (_xs == null || _xs.Length < n) _xs = new float[n];
            if (_ys == null || _ys.Length < n) _ys = new float[n];
            if (_zs == null || _zs.Length < n) _zs = new float[n];

            ReadFloatArrayInto(_xs, n);
            ReadFloatArrayInto(_ys, n);
            ReadFloatArrayInto(_zs, n);

            for (int i = 0; i < n; i++)
                positions[i] = new Vector3(_xs[i], _ys[i], _zs[i]);

            if (HasAggregateSizes)
            {
                if (_aggBuf == null || _aggBuf.Length < n) _aggBuf = new ushort[n];
                ReadUShortArrayInto(_aggBuf, n);
                if (aggSizes != null)
                {
                    Array.Copy(_aggBuf, aggSizes, n);
                }
            }
            else if (aggSizes != null)
            {
                // v1 file with no aggregate info — treat all as singletons.
                for (int i = 0; i < n; i++) aggSizes[i] = 1;
            }

            return time;
        }

        // ----------------------------------------------------------------
        // IDisposable
        // ----------------------------------------------------------------

        public void Dispose()
        {
            _br?.Dispose();
            _fs?.Dispose();
        }

        // ----------------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------------

        private void ReadFloatArrayInto(float[] dest, int count)
        {
            int byteCount = count * sizeof(float);
            if (_readBuf.Length < byteCount)
                _readBuf = new byte[byteCount];

            int read = 0;
            while (read < byteCount)
                read += _fs.Read(_readBuf, read, byteCount - read);

            Buffer.BlockCopy(_readBuf, 0, dest, 0, byteCount);
        }

        private void ReadUShortArrayInto(ushort[] dest, int count)
        {
            int byteCount = count * sizeof(ushort);
            if (_readBuf.Length < byteCount)
                _readBuf = new byte[byteCount];

            int read = 0;
            while (read < byteCount)
                read += _fs.Read(_readBuf, read, byteCount - read);

            Buffer.BlockCopy(_readBuf, 0, dest, 0, byteCount);
        }
    }
}
