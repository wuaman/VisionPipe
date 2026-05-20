#pragma once

#include <array>
#include <cmath>
#include <cstring>

namespace visionpipe {

struct TrackBox;

class KalmanBoxTracker {
public:
    static constexpr int N = 7;
    static constexpr int M = 4;

    explicit KalmanBoxTracker(float cx, float cy, float s, float r) {
        std::memset(x_, 0, sizeof(x_));
        x_[0] = cx;
        x_[1] = cy;
        x_[2] = s;
        x_[3] = r;

        identity(P_);
        P_[0][0] = 10.f; P_[1][1] = 10.f; P_[2][2] = 10.f; P_[3][3] = 10.f;
        P_[4][4] = 1e4f; P_[5][5] = 1e4f; P_[6][6] = 1e4f;

        init_matrices();
    }

    void predict() {
        // x = F * x
        float nx[N];
        mat_vec_mul(F_, x_, nx);
        std::memcpy(x_, nx, sizeof(x_));

        // P = F * P * F^T + Q
        float FP[N][N], FPFT[N][N];
        mat_mul(F_, P_, FP);
        mat_mul_transpose(FP, F_, FPFT);
        mat_add(FPFT, Q_, P_);

        // Clamp area to prevent negative
        if (x_[2] < 1e-6f) x_[2] = 1e-6f;
    }

    void update(float cx, float cy, float s, float r) {
        float z[M] = {cx, cy, s, r};

        // y = z - H * x (innovation)
        float Hx[M];
        for (int i = 0; i < M; ++i) {
            Hx[i] = 0;
            for (int j = 0; j < N; ++j) Hx[i] += H_[i][j] * x_[j];
        }
        float y[M];
        for (int i = 0; i < M; ++i) y[i] = z[i] - Hx[i];

        // S = H * P * H^T + R
        float PH[N][M];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                PH[i][j] = 0;
                for (int k = 0; k < N; ++k) PH[i][j] += P_[i][k] * H_[j][k];
            }
        float S[M][M];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j) {
                S[i][j] = R_[i][j];
                for (int k = 0; k < N; ++k) S[i][j] += H_[i][k] * PH[k][j];
            }

        // K = P * H^T * S^{-1}
        float S_inv[M][M];
        invert4x4(S, S_inv);

        float K[N][M];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) {
                K[i][j] = 0;
                for (int k = 0; k < M; ++k) K[i][j] += PH[i][k] * S_inv[k][j];
            }

        // x = x + K * y
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) x_[i] += K[i][j] * y[j];

        // P = (I - K * H) * P
        float KH[N][N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                KH[i][j] = 0;
                for (int k = 0; k < M; ++k) KH[i][j] += K[i][k] * H_[k][j];
            }
        float IKH[N][N];
        identity(IKH);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) IKH[i][j] -= KH[i][j];

        float newP[N][N];
        mat_mul(IKH, P_, newP);
        std::memcpy(P_, newP, sizeof(P_));
    }

    void get_state(float& cx, float& cy, float& s, float& r) const {
        cx = x_[0];
        cy = x_[1];
        s = std::max(x_[2], 1e-6f);
        r = x_[3];
    }

    void get_bbox(float bbox[4]) const {
        float cx = x_[0], cy = x_[1];
        float s = std::max(x_[2], 1e-6f), r = x_[3];
        float w = std::sqrt(s * r);
        float h = s / std::max(w, 1e-6f);
        bbox[0] = cx - w / 2;
        bbox[1] = cy - h / 2;
        bbox[2] = cx + w / 2;
        bbox[3] = cy + h / 2;
    }

    static void bbox_to_xysr(const float bbox[4], float& cx, float& cy, float& s, float& r) {
        float w = bbox[2] - bbox[0];
        float h = bbox[3] - bbox[1];
        cx = bbox[0] + w / 2;
        cy = bbox[1] + h / 2;
        s = w * h;
        r = (h > 1e-6f) ? w / h : 1.0f;
    }

private:
    float x_[N];
    float P_[N][N];
    float F_[N][N];
    float H_[M][N];
    float Q_[N][N];
    float R_[M][M];

    void init_matrices() {
        // F: state transition (constant velocity)
        identity(F_);
        F_[0][4] = 1.f;  // cx += vx
        F_[1][5] = 1.f;  // cy += vy
        F_[2][6] = 1.f;  // s  += vs

        // H: measurement matrix (observe [cx, cy, s, r])
        std::memset(H_, 0, sizeof(H_));
        for (int i = 0; i < M; ++i) H_[i][i] = 1.f;

        // Q: process noise (standard ByteTrack values)
        std::memset(Q_, 0, sizeof(Q_));
        float std_pos = 1.f / 20.f;
        float std_vel = 1.f / 160.f;
        Q_[0][0] = std_pos * std_pos;
        Q_[1][1] = std_pos * std_pos;
        Q_[2][2] = 1e-2f;
        Q_[3][3] = std_pos * std_pos;
        Q_[4][4] = std_vel * std_vel;
        Q_[5][5] = std_vel * std_vel;
        Q_[6][6] = 1e-4f;

        // R: measurement noise
        std::memset(R_, 0, sizeof(R_));
        float std_meas = 1.f / 20.f;
        R_[0][0] = std_meas * std_meas;
        R_[1][1] = std_meas * std_meas;
        R_[2][2] = 1e-1f;
        R_[3][3] = std_meas * std_meas;
    }

    static void identity(float m[N][N]) {
        std::memset(m, 0, N * N * sizeof(float));
        for (int i = 0; i < N; ++i) m[i][i] = 1.f;
    }

    static void mat_vec_mul(const float A[N][N], const float v[N], float out[N]) {
        for (int i = 0; i < N; ++i) {
            out[i] = 0;
            for (int j = 0; j < N; ++j) out[i] += A[i][j] * v[j];
        }
    }

    static void mat_mul(const float A[N][N], const float B[N][N], float C[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < N; ++k) C[i][j] += A[i][k] * B[k][j];
            }
    }

    static void mat_mul_transpose(const float A[N][N], const float B[N][N], float C[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < N; ++k) C[i][j] += A[i][k] * B[j][k];
            }
    }

    static void mat_add(const float A[N][N], const float B[N][N], float C[N][N]) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) C[i][j] = A[i][j] + B[i][j];
    }

    static void invert4x4(const float m[M][M], float inv[M][M]) {
        float det;
        float a[16], b[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) a[i * 4 + j] = m[i][j];

        b[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
        b[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
        b[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
        b[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];

        b[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
        b[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
        b[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
        b[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];

        b[2]  =  a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
        b[6]  = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
        b[10] =  a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
        b[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];

        b[3]  = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
        b[7]  =  a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
        b[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9]  + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
        b[15] =  a[0]*a[5]*a[10] - a[0]*a[6]*a[9]  - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];

        det = a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12];
        if (std::abs(det) < 1e-12f) {
            std::memset(inv, 0, sizeof(float) * M * M);
            return;
        }
        float inv_det = 1.0f / det;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) inv[i][j] = b[i * 4 + j] * inv_det;
    }
};

}  // namespace visionpipe
