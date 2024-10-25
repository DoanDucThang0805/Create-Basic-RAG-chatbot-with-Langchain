# Xây dựng hệ thống RAG cơ bản với Langchain


Trong dự án này, RAG LLM chatbot đã được phát triển dựa trên Langchain với LLM gen token là gemini

## Hướng dẫn sử dụng
* Để có thể chạy được project này yêu cầu bắt buộc bạn phải có:
  * Môi trường anaconda đã được kích hoạt
  * Python >= 3.9
  * IDE : Pycharm, VScode
  * API key từ gemini (hãy truy cập vào https://aistudio.google.com/app/prompts/new_chat để khởi tạo API key của bạn)
  * Sau đó bạn hãy tạo file .env để thêm biến **API_KEY = "your_API_key"**
  * Cài đặt các thư viện cần thiết có trong file requirements.txt bằng cách gõ : **pip install -r requirements.txt**
  
* Sau khi đã setup và cài đặt thành công, bạn có thể cho chatbot học các tài liệu của bạn bằng cách
  * Hãy tạo thư mục data tại nơi dự án của bạn
  * Cho thêm các tài liệu mà bạn mong muốn
  * Lưu ý : Hãy đưa các tài liệu có định dạng pdf
  * Cuối cùng bạn hãy chạy file prepare_vector.py để lưu trữ thông tin trong tài liệu dưới dạng vector

* Cuối cùng, để sử dụng ứng dụng, bạn hãy gõ lệnh **streamlit run qabot.py**

## Hy vọng đây sẽ là tài liệu tham khảo hữu ích cho người mới bắt đầu. Xin cảm ơn !!!