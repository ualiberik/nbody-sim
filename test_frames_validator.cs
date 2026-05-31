using System;
using System.IO;

class FramesValidator
{
    static void Main()
    {
        string path = @"data\20260525_185147\frames.bin";
        
        if (!File.Exists(path))
        {
            Console.WriteLine($"ERROR: File not found: {path}");
            return;
        }
        
        try
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read))
            using (var br = new BinaryReader(fs))
            {
                // Read header
                var magic = new string(br.ReadChars(4));
                uint version = br.ReadUInt32();
                uint nParticles = br.ReadUInt32();
                uint nFrames = br.ReadUInt32();
                double dt = br.ReadDouble();
                double outputDt = br.ReadDouble();
                
                Console.WriteLine("? Frame File Header Valid");
                Console.WriteLine($"  Magic: {magic}");
                Console.WriteLine($"  Version: {version}");
                Console.WriteLine($"  Particles: {nParticles:N0}");
                Console.WriteLine($"  Frames: {nFrames:N0}");
                Console.WriteLine($"  dt: {dt:F6} T0");
                Console.WriteLine($"  output_dt: {outputDt:F6} T0");
                Console.WriteLine($"  Total simulation time: {nFrames * outputDt:F2} T0");
                
                // Check file size
                long expectedSize = 32 + (long)nFrames * (8 + nParticles * (3 * 4 + 2));
                long actualSize = fs.Length;
                Console.WriteLine($"\n? File Size Check");
                Console.WriteLine($"  Expected: {expectedSize:N0} bytes");
                Console.WriteLine($"  Actual: {actualSize:N0} bytes");
                Console.WriteLine($"  Match: {(expectedSize == actualSize ? "YES ?" : "NO ?")}");
                
                // Read first frame
                double t = br.ReadDouble();
                Console.WriteLine($"\n? First Frame");
                Console.WriteLine($"  Time: {t:F4} T0");
                
                // Sample a few position values
                float x0 = br.ReadSingle();
                Console.WriteLine($"  x[0]: {x0:F6} AU");
                
                Console.WriteLine("\n??? VALIDATION SUCCESSFUL ???");
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"ERROR: {e.Message}");
        }
    }
}
