import pandas as pd
from sklearn.preprocessing import StandardScaler

## 메타데이터 처리: 결측치 처리, 범주형 변수 인코딩, 수치형 변수 인코딩 ##
def preprocess_metadata(df: pd.DataFrame, scaler=None, fit_scaler=True):
    # 결측치 처리
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    df['sex'] = df['sex'].fillna('unknown')
    df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')
    
    # 범주형 변수 one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['sex', 'anatom_site_general'], drop_first=True)
    
    # 수치형 변수 스케일링
    # 수치형 변수만 선택
    numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['target']]
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    elif scaler is not None:
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
        
    return df_encoded, scaler


## 이미지 증강 정의 ##
def get_transform(phase="train"):
    from torchvision import transforms
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])