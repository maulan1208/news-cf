## Cấu trúc
news/
├── config.py            Đường dẫn dữ liệu, tham số dùng chung 
├── preprocess.py        Đọc data -> ma trận thưa user × news
├── item.py              Model Item: chuẩn hoá vector + cosine similarity
├── recommend.py         Demo gợi ý theo history
├── evaluate.py          Đánh giá: AUC, MRR, nDCG@5, nDCG@10 
├── results.json         Kết quả đánh giá 
├── README.md
└── data/
    ├── raw/       
    │   ├── MINDsmall_train/   
    │   └── MINDsmall_dev/     
    └── processed/       
        ├── user_item.npz      
        ├── user2idx.pkl       
        └── news2idx.pkl       

## Nguyên lý model (Item-based)
- Biểu diễn tin tức: mỗi tin là một vector trên không gian user - vector[i] = 1 nếu user i đã đọc tin đó, ngược lại 0. Hai tin được cùng một nhóm user đọc thì vector của chúng gần nhau → giống nhau.
- Độ giống nhau giữa hai tin = cosine similarity giữa hai vector. Vì mọi vector tin đều được L2-normaliz (đưa về độ dài 1) trong fit(), nên dot product giữa chúng chính là cosine similarity.
- Tránh ma trận item-item khổng lồ: thay vì tính sẵn ma trận similarity n_items × n_items (tốn bộ nhớ), gom history của user thành một profile vector rồi so từng tin candidate với profile đó.
