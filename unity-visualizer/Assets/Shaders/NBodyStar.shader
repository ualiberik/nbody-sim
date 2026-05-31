// NBody/Star
// Renders a point-mesh particle as a glowing, pulsating star with corona.
// Geometry shader expands the point into TWO screen-aligned quads:
//   - inner quad (size 1×)  : solid hot surface with hot/cool radial gradient
//   - outer quad (size 4×)  : soft corona / halo (additive falloff)
// Both write a single point but produce 8 vertices = two quads in one pass.
//
// Used for the central host star: bright, warm, alive.

Shader "NBody/Star"
{
    Properties
    {
        _CoreSize       ("Core diameter (AU)",   Float) = 0.5
        _CoronaSize     ("Corona diameter (AU)", Float) = 3.0
        _CoreColor      ("Core color",   Color) = (1.0, 0.95, 0.6, 1)
        _HotColor       ("Centre hot color", Color) = (1.5, 1.2, 0.8, 1)
        _CoronaColor    ("Corona color", Color) = (1.0, 0.6, 0.2, 1)
        _Brightness     ("Brightness",   Range(0.5, 5)) = 2.2
        _PulseAmplitude ("Pulse amount", Range(0, 0.5)) = 0.08
        _PulseSpeed     ("Pulse speed",  Range(0, 5))   = 1.5
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent+10" }
        Blend SrcAlpha One                       // additive — hides hard edges
        ZWrite Off                               // draw on top of dust
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma target   4.0
            #pragma vertex   vert
            #pragma geometry geo
            #pragma fragment frag
            #include "UnityCG.cginc"

            float  _CoreSize, _CoronaSize;
            float4 _CoreColor, _HotColor, _CoronaColor;
            float  _Brightness, _PulseAmplitude, _PulseSpeed;

            struct appdata { float4 vertex : POSITION; };
            struct v2g     { float4 wpos   : TEXCOORD0; };
            struct g2f
            {
                float4 clipPos : SV_POSITION;
                float2 uv      : TEXCOORD0;   // [-1,1] across each quad
                float  layer   : TEXCOORD1;   // 0=core, 1=corona
            };

            v2g vert(appdata v)
            {
                v2g o; o.wpos = mul(unity_ObjectToWorld, v.vertex); return o;
            }

            // -------- Emit a screen-aligned quad in view space --------
            void emitQuad(inout TriangleStream<g2f> stream,
                           float3 viewCenter, float halfSize, float layer)
            {
                g2f o; o.layer = layer;
                float3 c0 = viewCenter + float3(-halfSize, -halfSize, 0);
                float3 c1 = viewCenter + float3( halfSize, -halfSize, 0);
                float3 c2 = viewCenter + float3(-halfSize,  halfSize, 0);
                float3 c3 = viewCenter + float3( halfSize,  halfSize, 0);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c0, 1)); o.uv = float2(-1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c1, 1)); o.uv = float2( 1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c2, 1)); o.uv = float2(-1, 1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c3, 1)); o.uv = float2( 1, 1); stream.Append(o);
                stream.RestartStrip();
            }

            [maxvertexcount(8)]
            void geo(point v2g input[1], inout TriangleStream<g2f> stream)
            {
                float pulse = 1.0 + _PulseAmplitude * sin(_Time.y * _PulseSpeed * 6.2831);

                float4 vp = mul(UNITY_MATRIX_V, float4(input[0].wpos.xyz, 1.0));

                // First: corona (drawn first so core overlays it).
                emitQuad(stream, vp.xyz, _CoronaSize * 0.5 * pulse, 1.0);
                // Then: hot core
                emitQuad(stream, vp.xyz, _CoreSize   * 0.5 * pulse, 0.0);
            }

            fixed4 frag(g2f i) : SV_Target
            {
                float r = length(i.uv);
                if (r > 1.0) discard;

                if (i.layer < 0.5) {
                    // ── Hot CORE — solid disc with hot-to-rim falloff
                    float t = saturate(r);
                    // centre is hottest, edges cooler
                    float3 col = lerp(_HotColor.rgb, _CoreColor.rgb, t * t);
                    // Bright edge bloom — small sigmoid lift near r=1
                    col += _HotColor.rgb * 0.2 * smoothstep(0.7, 1.0, t);
                    return fixed4(col * _Brightness, 1.0 - t * t);
                } else {
                    // ── CORONA — radial gradient, additive blending
                    // Falloff: hottest at small r, fades to 0 at r=1
                    float falloff = pow(saturate(1.0 - r), 2.5);
                    // Asymmetric radial flare: a bit hotter near the disc
                    falloff += 0.3 * pow(saturate(1.0 - r), 6.0);
                    float3 col = _CoronaColor.rgb * falloff * _Brightness;
                    return fixed4(col, falloff);
                }
            }
            ENDCG
        }
    }
    Fallback "Unlit/Color"
}
