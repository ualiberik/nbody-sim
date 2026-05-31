// NBody/Perturber
// Renders the fixed outer gas-giant perturber as a single glowing sphere.
// Same hemispherical-sphere trick as NBody/Particle but with a brighter,
// colour-shifted appearance and a tiny "trail" line that hints at its
// circular orbit (purely visual, doesn't reflect the actual orbit).

Shader "NBody/Perturber"
{
    Properties
    {
        _Diameter   ("Diameter (AU)", Float)        = 1.2
        _BodyColor  ("Body color",    Color)        = (0.55, 0.85, 0.55, 1)
        _GlowColor  ("Glow color",    Color)        = (0.30, 1.00, 0.40, 1)
        _Brightness ("Brightness",    Range(1,8))   = 3.0
        _Ambient    ("Ambient",       Range(0,1))   = 0.4
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry+5" }
        ZWrite On
        ZTest LEqual
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma target   4.0
            #pragma vertex   vert
            #pragma geometry geo
            #pragma fragment frag
            #include "UnityCG.cginc"

            float  _Diameter;
            float4 _BodyColor, _GlowColor;
            float  _Brightness, _Ambient;

            struct appdata { float4 vertex : POSITION; };
            struct v2g     { float4 wpos   : TEXCOORD0; };
            struct g2f
            {
                float4 clipPos : SV_POSITION;
                float2 uv      : TEXCOORD0;
            };

            v2g vert(appdata v)
            {
                v2g o; o.wpos = mul(unity_ObjectToWorld, v.vertex); return o;
            }

            [maxvertexcount(4)]
            void geo(point v2g input[1], inout TriangleStream<g2f> stream)
            {
                float3 vp = mul(UNITY_MATRIX_V, float4(input[0].wpos.xyz, 1)).xyz;
                float  hs = _Diameter * 0.5;

                g2f o;
                float3 c0 = vp + float3(-hs, -hs, 0);
                float3 c1 = vp + float3( hs, -hs, 0);
                float3 c2 = vp + float3(-hs,  hs, 0);
                float3 c3 = vp + float3( hs,  hs, 0);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c0,1)); o.uv = float2(-1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c1,1)); o.uv = float2( 1,-1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c2,1)); o.uv = float2(-1, 1); stream.Append(o);
                o.clipPos = mul(UNITY_MATRIX_P, float4(c3,1)); o.uv = float2( 1, 1); stream.Append(o);
            }

            fixed4 frag(g2f i) : SV_Target
            {
                float r2 = dot(i.uv, i.uv);
                if (r2 > 1.0) discard;

                // Sphere normal Z, Lambert shading toward camera.
                float nz = sqrt(1.0 - r2);
                float shade = _Ambient + (1.0 - _Ambient) * saturate(nz);

                // Inner body + outer rim glow.
                float rim = smoothstep(0.6, 1.0, sqrt(r2));
                float3 col = lerp(_BodyColor.rgb, _GlowColor.rgb, rim);

                return fixed4(col * shade * _Brightness, 1.0);
            }
            ENDCG
        }
    }
    Fallback "Unlit/Color"
}
