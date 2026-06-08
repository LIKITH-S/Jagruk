import 'package:flutter_bloc/flutter_bloc.dart';

import '../services/news_service.dart';

import 'news_event.dart';
import 'news_state.dart';

class NewsBloc extends Bloc<NewsEvent, NewsState> {
  final NewsService service;

  NewsBloc(this.service) : super(NewsLoading()) {
    on<LoadNews>((event, emit) async {
      try {
        final news = await service.getNews();

        emit(NewsLoaded(news));
      } catch (e) {
        emit(NewsError("Failed to load news"));
      }
    });
  }
}
