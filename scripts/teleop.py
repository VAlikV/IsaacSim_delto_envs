import cv2
import mediapipe as mp
import numpy as np
import socket

def calcAngle(p1, p2, p3, p4):
    v1 = p2-p1
    v2 = p4-p3

    angle = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

    return np.arccos(angle)

names = {
        "rj_dg_1_1" : 0,
        "rj_dg_1_2" : 1,
        "rj_dg_1_3" : 2,
        "rj_dg_1_4" : 3,
        "rj_dg_2_1" : 4,
        "rj_dg_2_2" : 5,
        "rj_dg_2_3" : 6,
        "rj_dg_2_4" : 7,
        "rj_dg_3_1" : 8,
        "rj_dg_3_2" : 9,
        "rj_dg_3_3" : 10,
        "rj_dg_3_4" : 11,
        "rj_dg_4_1" : 12,
        "rj_dg_4_2" : 13,
        "rj_dg_4_3" : 14,
        "rj_dg_4_4" : 15,
        "rj_dg_5_1" : 16,
        "rj_dg_5_2" : 17,
        "rj_dg_5_3" : 18,
        "rj_dg_5_4" : 19,
        }

joints = {
        "rj_dg_1_1":[0,1,1,2],
        "rj_dg_1_3":[1,2,2,3],
        "rj_dg_1_4":[2,3,3,4],

        "rj_dg_2_2":[0,5,5,6],
        "rj_dg_2_3":[5,6,6,7],
        "rj_dg_2_4":[6,7,7,8],

        "rj_dg_3_2":[0,9,9,10],
        "rj_dg_3_3":[9,10,10,11],
        "rj_dg_3_4":[10,11,11,12],
        
        "rj_dg_4_2":[0,13,13,14],
        "rj_dg_4_3":[13,14,14,15],
        "rj_dg_4_4":[14,15,15,16],

        "rj_dg_5_3":[17,18,18,19],
        "rj_dg_5_4":[18,19,19,20]
        }

angles = {
        "rj_dg_1_1" : 0.0,
        "rj_dg_1_2" : -0.01,
        "rj_dg_1_3" : 0.0,
        "rj_dg_1_4" : 0.0,

        "rj_dg_2_2" : 0.0,
        "rj_dg_2_3" : 0.0,
        "rj_dg_2_4" : 0.0,

        "rj_dg_3_2" : 0.0,
        "rj_dg_3_3" : 0.0,
        "rj_dg_3_4" : 0.0,
                
        "rj_dg_4_2" : 0.0,
        "rj_dg_4_3" : 0.0,
        "rj_dg_4_4" : 0.0,

        "rj_dg_5_3" : 0.0,
        "rj_dg_5_4" : 0.0,
        }

# order = ["WRJ1","WRJ0","FFJ3","FFJ2","FFJ1","LFJ4","LFJ3","LFJ2","LFJ1","RFJ3","RFJ2","RFJ1","MFJ3","MFJ2","MFJ1","THJ4","THJ3","THJ2","THJ1","THJ0"]

order = ["rj_dg_1_1",
         "rj_dg_2_1",
         "rj_dg_3_1",
         "rj_dg_4_1",
         "rj_dg_5_2",
         "rj_dg_1_2",
         "rj_dg_2_2",
         "rj_dg_3_2",
         "rj_dg_4_2",
         "rj_dg_5_2",
         "rj_dg_1_3",
         "rj_dg_2_3",
         "rj_dg_3_3",
         "rj_dg_4_3",
         "rj_dg_5_3",
         "rj_dg_1_4",
         "rj_dg_2_4",
         "rj_dg_3_4",
         "rj_dg_4_4",
         "rj_dg_5_4"]

message = [0] * 20

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 8080))
sock.settimeout(1.0)

count = 0

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # работает с видео
    max_num_hands=2,               # до двух рук
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

# Запуск камеры
cap = cv2.VideoCapture(0)  # 0 — основная камера

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ошибка при захвате изображения")
        break

    # Преобразование в RGB (требуется MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка изображения
    results = hands.process(frame_rgb)

    # Отображение результатов
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for k, v in joints.items():
                thumb_tip1 = hand_landmarks.landmark[v[0]]
                thumb_tip2 = hand_landmarks.landmark[v[1]]
                thumb_tip3 = hand_landmarks.landmark[v[3]]

                p1 = np.array([thumb_tip1.x, thumb_tip1.y, thumb_tip1.z])
                p2 = np.array([thumb_tip2.x, thumb_tip2.y, thumb_tip2.z])
                p3 = np.array([thumb_tip3.x, thumb_tip3.y, thumb_tip3.z])

                angles[k] = calcAngle(p1, p2, p2, p3)

                # cv2.putText(frame, str(v), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            l1 = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z]) - np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
            l2 = np.array([hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z]) - np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
            l3 = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z]) - np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])

            n = np.cross(l1, l2)

            angles["rj_dg_1_2"] = -(np.pi/2 - np.arccos(np.dot(l3,n)/(np.linalg.norm(l3)*np.linalg.norm(n))))

            for i, name in enumerate(order):
                if name in angles.keys():
                    message[i] = angles[name]

            # print(message)
            print(angles["rj_dg_1_2"]*180/np.pi)

            # ==================================================================================================================

            # print(message)
            if count >= 20:
                print("Sent")
                sock.sendto(str(message).encode('utf-8'), ("127.0.0.1", 8081))
                count = 0
            
            count += 1


    # Отображение окна
    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc — выход
        break

# Очистка
cap.release()
cv2.destroyAllWindows()
hands.close()
