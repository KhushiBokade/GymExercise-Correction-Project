import os
import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np
import time
import collections
import matplotlib.pyplot as plt

# ------------------ MODEL & POSE SETUP (UNCHANGED) ------------------ #

model = joblib.load("exercise_classifier_best.pkl")
le_label = joblib.load("label_encoder.pkl")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ------------------ ANGLE & FEATURE FUNCTIONS (UNCHANGED) ------------------ #

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def extract_pose_features(landmarks):
    if not landmarks or len(landmarks) < mp_pose.PoseLandmark.RIGHT_ANKLE.value+1:
        return None
    lmk = lambda name: [landmarks[mp_pose.PoseLandmark[name].value].x,
                        landmarks[mp_pose.PoseLandmark[name].value].y]
    a = lambda p1,p2,p3: calculate_angle(lmk(p1), lmk(p2), lmk(p3))

    features = pd.DataFrame([{
        'Shoulder_Angle': a('LEFT_HIP','LEFT_SHOULDER','LEFT_ELBOW'),
        'Elbow_Angle':    a('LEFT_SHOULDER','LEFT_ELBOW','LEFT_WRIST'),
        'Hip_Angle':      a('LEFT_SHOULDER','LEFT_HIP','LEFT_KNEE'),
        'Knee_Angle':     a('LEFT_HIP','LEFT_KNEE','LEFT_ANKLE'),
        'Ankle_Angle':    calculate_angle(
                              lmk('LEFT_KNEE'), lmk('LEFT_ANKLE'),
                              [lmk('LEFT_ANKLE')[0]+0.1, lmk('LEFT_ANKLE')[1]]),
        'Shoulder_Ground_Angle': a('RIGHT_HIP','RIGHT_SHOULDER','RIGHT_ELBOW'),
        'Elbow_Ground_Angle':    a('RIGHT_SHOULDER','RIGHT_ELBOW','RIGHT_WRIST'),
        'Hip_Ground_Angle':      a('RIGHT_SHOULDER','RIGHT_HIP','RIGHT_KNEE'),
        'Knee_Ground_Angle':     a('RIGHT_HIP','RIGHT_KNEE','RIGHT_ANKLE'),
        'Ankle_Ground_Angle':    calculate_angle(
                              lmk('RIGHT_KNEE'), lmk('RIGHT_ANKLE'),
                              [lmk('RIGHT_ANKLE')[0]+0.1, lmk('RIGHT_ANKLE')[1]]),
        'Torso_Angle': calculate_angle(
            lmk('LEFT_HIP'),
            [(lmk('LEFT_SHOULDER')[0]+lmk('RIGHT_SHOULDER')[0])/2,
             (lmk('LEFT_SHOULDER')[1]+lmk('RIGHT_SHOULDER')[1])/2],
            lmk('LEFT_KNEE'))
    }])
    return features

# ------------------ CLASSIFICATION & REP COUNT (UNCHANGED) ------------------ #

def classify_pose(features):
    if features is None or features.empty or features.shape[1]!=11:
        return "Unknown", 0.0
    try:
        pf = features.drop(columns=['Torso_Angle'])
        pred = model.predict(pf)[0]
        prob = model.predict_proba(pf)[0]
        conf = prob.max()
        if conf<0.4:
            ea = features['Elbow_Angle'].iloc[0]
            sa = features['Shoulder_Angle'].iloc[0]
            if 70<ea<150: return "push-up",0.4
            if 90<sa<180: return "pull-up",0.4
            return "Unknown",conf
        return le_label.inverse_transform([pred])[0],conf
    except Exception as e:
        print("Classification error:",e)
        return "Unknown",0.0

def count_repetitions(prev, curr, pa, ca):
    return 1 if prev and curr in ["push-up","pull-up"] and abs(ca-pa)>20 else 0

# ------------------ MAIN EXERCISE VIDEO PROCESSING (UNCHANGED) ------------------ #

def process_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening {source}")
        return

    cv2.namedWindow('Exercise Classification', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Exercise Classification',1280,720)

    prev_ex, prev_elb, reps = "Unknown",0,0
    frame_cnt, start = 0, time.time()
    queue = collections.deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_cnt+=1

        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec((245,117,66),2,4),
                mp_drawing.DrawingSpec((245,66,230),2,4))
            feats = extract_pose_features(res.pose_landmarks.landmark)
            ex,conf = classify_pose(feats)
            queue.append(ex)
            curr_elb = feats['Elbow_Angle'].iloc[0] if feats is not None else prev_elb
            stab = max(set(queue),key=queue.count) if len(queue)==queue.maxlen else prev_ex
            if stab in ["push-up","pull-up"]:
                reps += count_repetitions(prev_ex, stab, prev_elb, curr_elb)
                prev_elb = curr_elb
            prev_ex = stab

            cv2.putText(img,f'Exercise: {stab}',(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(img,f'Confidence: {conf:.2f}',(20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('Exercise Classification',img)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ ENHANCED POSTURE ANALYSIS PHASE ------------------ #

def analyze_posture(features):
    """
    Returns (feedback_str, is_correct_bool, error_codes_list)
    """
    if features is None or features.empty:
        return "No Pose Detected", False, ["no_landmarks"]
    errs = []
    ea = features['Elbow_Angle'].iloc[0]
    sa = features['Shoulder_Angle'].iloc[0]
    ta = features['Torso_Angle'].iloc[0]

    if ea<70 or ea>150:    errs.append("elbow_range")
    if sa<90 or sa>180:    errs.append("shoulder_align")
    if ta<70 or ta>160:    errs.append("torso_tilt")

    if not errs:
        return "Posture Correct", True, []
    txts = []
    if "elbow_range"    in errs: txts.append("Elbow out of range")
    if "shoulder_align" in errs: txts.append("Shoulder misaligned")
    if "torso_tilt"     in errs: txts.append("Torso tilt off")
    return " / ".join(txts), False, errs

def wrap_text(text, max_width, font, scale, thickness):
    """Utility to wrap text into multiple lines."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = cur + " " + w if cur else w
        (w_px,_),_ = cv2.getTextSize(test, font, scale, thickness)
        if w_px <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def playback_with_posture_analysis(source="pull.mp4"):
    out_dir = "results_posture-analysis"
    os.makedirs(out_dir, exist_ok=True)

    cap      = cv2.VideoCapture(source)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w,h      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')

    full_out = os.path.join(out_dir,"posture_full.mp4")
    err_out  = os.path.join(out_dir,"posture_errors.mp4")
    writer_f = cv2.VideoWriter(full_out, fourcc, fps, (w,h))
    writer_e = cv2.VideoWriter(err_out,  fourcc, fps, (w,h))

    log_rows = []
    stats    = collections.Counter()
    frame_i  = 0
    bar_h    = 10

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_i += 1
        t_sec    = frame_i / fps

        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # progress bar
        prog = int((frame_i/total_fr)*w)
        cv2.rectangle(img,(0,h-bar_h),(prog,h),(50,200,50),-1)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec((100,255,100),2,2),
                mp_drawing.DrawingSpec((100,100,255),2,2))
            feats = extract_pose_features(res.pose_landmarks.landmark)
            fb, ok, errs = analyze_posture(feats)
            stats["correct" if ok else "error"] += 1
            for e in errs: stats[e]+=1
        else:
            fb, ok, errs = "No Pose Detected", False, ["no_landmarks"]
            stats["no_landmarks"] +=1

        # wrap feedback text
        full_txt = f"{t_sec:5.2f}s | {fb}"
        lines = wrap_text(full_txt, w-40, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        color = (0,255,0) if ok else (0,0,255)
        # background box
        box_h = 30 * len(lines)
        cv2.rectangle(img,(10,10),(w-10,10+box_h),(0,0,0),-1)
        # put lines
        for idx, line in enumerate(lines):
            y = 10 + 25*(idx+1)
            cv2.putText(img,line,(15,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        writer_f.write(img)
        if not ok:
            writer_e.write(img)

        log_rows.append({
            "frame": frame_i,
            "time_s": round(t_sec,2),
            "feedback": fb,
            "errors": ";".join(errs)
        })

        cv2.imshow('Posture Analysis',img)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    writer_f.release()
    writer_e.release()
    cv2.destroyAllWindows()

    # save log
    df = pd.DataFrame(log_rows)
    csv_path = os.path.join(out_dir,"posture_log.csv")
    df.to_csv(csv_path,index=False)

    # overall percentages
    total = frame_i
    corr  = stats['correct']
    errf  = stats['error']
    pct_corr = corr/total*100
    pct_err  = errf/total*100

    # improvement suggestions
    suggestions = []
    for code, cnt in stats.most_common():
        if code in ("correct","error","no_landmarks"): continue
        if code=="elbow_range":
            suggestions.append("Work on keeping your elbows within the proper bend.")
        if code=="shoulder_align":
            suggestions.append("Try to align your shoulders square to the camera.")
        if code=="torso_tilt":
            suggestions.append("Keep your torso more upright.")
    if not suggestions:
        overall_fb = "Excellent form throughout!"
    else:
        overall_fb = " & ".join(suggestions)

    # save summary file
    summary_path = os.path.join(out_dir,"posture_summary.txt")
    with open(summary_path,"w") as f:
        f.write(f"Total frames processed: {total}\n")
        f.write(f"Correct posture: {corr} frames ({pct_corr:.1f}%)\n")
        f.write(f"Incorrect posture: {errf} frames ({pct_err:.1f}%)\n\n")
        f.write("What can be improved:\n")
        f.write(overall_fb + "\n")

    # print summary
    print("\n=== Posture Analysis Complete ===")
    print(f"Full annotated video: {full_out}")
    print(f"Error-only video:    {err_out}")
    print(f"Log CSV:             {csv_path}")
    print(f"Summary file:        {summary_path}")
    print(overall_fb)

# ------------------ PERFORMANCE GRAPH (UNCHANGED) ------------------ #

def plot_performance():
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    models = ['XGBoost','RandomForest']
    classes = ['pull Up','push-up']
    accuracies = [0.97578629434673,0.97578629434673]
    precision = {'XGBoost':[0.98,0.98],'RandomForest':[0.98,0.98]}
    recall    = {'XGBoost':[0.95,0.99],'RandomForest':[0.95,0.99]}
    f1_score  = {'XGBoost':[0.96,0.98],'RandomForest':[0.96,0.98]}

    fig, ax = plt.subplots(figsize=(12,6))
    bw = 0.2
    idx = np.arange(len(classes))
    xo = np.array([-1.5,-0.5,0.5,1.5])*bw

    ax.bar(idx+xo[0],precision['XGBoost'],bw,label='XGB Prec')
    ax.bar(idx+xo[1],recall['XGBoost'],bw,label='XGB Rec')
    ax.bar(idx+xo[2],f1_score['XGBoost'],bw,label='XGB F1')
    ax.bar(idx+xo[0],precision['RandomForest'],bw,alpha=0.7,label='RF Prec')
    ax.bar(idx+xo[1],recall['RandomForest'],bw,alpha=0.7,label='RF Rec')
    ax.bar(idx+xo[2],f1_score['RandomForest'],bw,alpha=0.7,label='RF F1')

    for i,acc in enumerate(accuracies):
        ax.axhline(y=acc,xmin=(i-0.4)/2,xmax=(i+0.4)/2,linestyle='--',color='purple')
        ax.text(i,acc+0.005,f'{acc:.3f}',ha='center')

    ax.set_xticks(idx)
    ax.set_xticklabels(classes)
    ax.set_ylim(0.9,1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance')
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),ncol=3)
    fig.tight_layout()
    pp = os.path.join(result_dir,"performance_comparison.png")
    plt.savefig(pp,bbox_inches='tight')
    plt.close()
    print(f"Performance graph saved: {pp}")

# ------------------ RUN EVERYTHING ------------------ #

if __name__ == "__main__":
    video_path = "pull.mp4"  # or 0 for webcam
    process_video(video_path)
    playback_with_posture_analysis(video_path)
    plot_performance()
