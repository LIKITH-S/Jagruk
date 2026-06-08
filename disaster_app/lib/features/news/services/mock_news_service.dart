import '../models/news_model.dart';

import 'news_service.dart';

class MockNewsService implements NewsService {
  @override
  Future<List<NewsModel>> getNews() async {
    await Future.delayed(const Duration(seconds: 2));

    return [
      NewsModel(
        id: '1',

        title: 'Cyclone Warning Issued',

        description: 'Authorities preparing evacuation near coastal regions.',

        imageUrl:
            'https://images.unsplash.com/photo-1500375592092-40eb2168fd21',

        publishedAt: DateTime.now(),
      ),

      NewsModel(
        id: '2',

        title: 'Wildfire Spreading Rapidly',

        description: 'Emergency teams deployed to contain wildfire.',

        imageUrl:
            'https://images.unsplash.com/photo-1473448912268-2022ce9509d8',

        publishedAt: DateTime.now(),
      ),

      NewsModel(
        id: '3',

        title: 'Flood Alert In Multiple Areas',

        description: 'Heavy rainfall expected over the next 48 hours.',

        imageUrl:
            'https://images.unsplash.com/photo-1515694346937-94d85e41e6f0',

        publishedAt: DateTime.now(),
      ),
    ];
  }
}
