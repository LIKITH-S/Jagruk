import '../models/news_model.dart';

abstract class NewsService {
  Future<List<NewsModel>> getNews();
}
