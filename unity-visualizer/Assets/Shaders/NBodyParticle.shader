// NBody/Particle
// Renders a point-cloud mesh (MeshTopology.Points) as shaded 3D spheres.
// A geometry shader expands each point into a view-space aligned quad,
// and the fragment shader uses the quad UVs to fake a hemispherical normal,
// giving each particle the appearance of a solid sphere with hard edges.
//
// Opaque rendering with depth writes — particles occlude each other correctly,
// so dense clumps look like real 3D objects rather than additive smudges.
//
// Compatible with Unity's Built-in Render Pipeline.

Shader "NBody/Particle"
{
    Properties
    {
        [HideInInspector] _ParticleSize   ("Particle Size (AU)", Float)  = 0.06
        [HideInInspector] _InnerColor     ("Inner Color",  Color) = (1.0, 0.75, 0.2, 1)
        [HideInInspector] _OuterColor     ("Outer Color",  Color) = (0.2, 0.5,  1.0, 1)
        [HideInInspector] _ColorRadiusMax ("Color Radius Max", Float) = 5.0
        _Brightness ("Brightness", Range(0.1, 5.0)) = 1.4
        _Ambient    ("Ambient",    Range(0.0, 1.0)) = 0.25
    }

    SubShader
    {
        // ── Render state: OPAQUE spheres with depth writes ────────────
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        ZWrite On
        ZTest LEqual
        Cull Off                  // Geo-shader quads sit in the view-Z plane,
                                  //   so their normal points +Z (away from
                                  //   the camera in Unity view space).
                                  //   Cull Back would discard every quad.

        Pass
        {
            CGPROGRAM
            #pragma target   4.0            // Geometry shader requires SM 4.0
            #pragma vertex   vert
            #pragma geometry geo
            #pragma fragment frag
            #include "UnityCG.cginc"

            // ── Uniforms ──────────────────────────────────────────────
            float  _ParticleSize;
            float4 _InnerColor;
            float4 _OuterColor;
            float  _ColorRadiusMax;
            float  _Brightness;
            float  _Ambient;

            // ── Structs ───────────────────────────────────────────────

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color  : COLOR;
            };

            struct v2g
            {
                float4 worldPos : TEXCOORD0;
                float4 color    : COLOR;
            };

            struct g2f
            {
                float4 clipPos : SV_POSITION;
                float4 color   : COLOR;
                float2 uv      : TEXCOORD0;  // [-1,1] local quad coords
            };

            // ── Vertex shader ─────────────────────────────────────────
            v2g vert(appdata v)
            {
                v2g o;
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                o.color    = v.color;
                return o;
            }

            // ── Geometry shader ───────────────────────────────────────
            // Expand each world-space point into a screen-aligned quad in
            // view space. Pass the view-space center + radius so the
            // fragment shader can reconstruct a per-pixel depth on the sphere.
            [maxvertexcount(4)]
            void geo(point v2g input[1], inout TriangleStream<g2f> stream)
            {
                float3 worldPos = input[0].worldPos.xyz;
                float4 color    = input[0].color;

                float4 viewPos = mul(UNITY_MATRIX_V, float4(worldPos, 1.0));
                float  hs      = _ParticleSize * 0.5;

                float3 c0 = viewPos.xyz + float3(-hs, -hs, 0);
                float3 c1 = viewPos.xyz + float3( hs, -hs, 0);
                float3 c2 = viewPos.xyz + float3(-hs,  hs, 0);
                float3 c3 = viewPos.xyz + float3( hs,  hs, 0);

                g2f o;
                o.color = color;

                o.clipPos = mul(UNITY_MATRIX_P, float4(c0, 1)); o.uv = float2(-1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c1, 1)); o.uv = float2( 1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c2, 1)); o.uv = float2(-1, 1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c3, 1)); o.uv = float2( 1, 1); stream.Append(o);

                stream.RestartStrip();
            }

            // ── Fragment shader ───────────────────────────────────────
            // Treat the [-1,1] uv as a projection onto a unit sphere; recover
            // sqrt(1-r²) as the outward Z-component of the surface normal and
            // shade with simple Lambert against the view direction. Pixels
            // outside the unit disc are discarded for hard circular edges.
            fixed4 frag(g2f i) : SV_Target
            {
                float r2 = dot(i.uv, i.uv);
                if (r2 > 1.0) discard;

                float nz      = sqrt(1.0 - r2);
                float lambert = saturate(nz);
                float shade   = _Ambient + (1.0 - _Ambient) * lambert;

                // Vertex color alpha encodes per-particle aggregate brightness:
                //   1.0 = large clump (full brightness)
                //   <1  = small clump or singleton (dimmed)
                // Multiplying RGB by this factor preserves opaque rendering
                // (no transparency, no sorting issues) while still giving a
                // strong visual contrast between planets and background dust.
                float aggFactor = i.color.a;

                return fixed4(i.color.rgb * shade * _Brightness * aggFactor, 1.0);
            }
            ENDCG
        }
    }

    Fallback "Unlit/Color"
}
