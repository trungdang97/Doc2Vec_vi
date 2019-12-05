# Doc2Vec_vi

Sử dụng thư viện vnTokenizer-4.1.1 (non-Spark) của Lê Hồng Phương (https://github.com/vuthaihoc/vntokenizer4.1) để tách từ theo file.
Có thể build lại sử dụng API tách theo string input.

Đường dẫn corpus sử dụng: https://github.com/binhvq/news-corpus
Chú ý corpus có kích thước rất lớn (> 600mb) nên cần có đủ bộ nhớ mở file.

Corpus có dạng các câu được tokenize như file "datasets/vi/corpus-title-small.tok.txt".
Tokenize file bằng CMD: java -jar <tên_file_executable_jar> -i "<đường_dẫn_file_input>" -o "<đường_dẫn_file_output>"

File train: doc2vec_vi/doc2vec_train.py (sửa đường dẫn đến corpus)
File test: doc2vec_vi/doc2vec_test.py

Sử dụng infer_vector() để lấy vector từ list từ đã được tách.
Sử dụng spatial.distance.cosine để tính cosine distance giữa 2 vector: cosine_distance = 1 - cosine_similarity
