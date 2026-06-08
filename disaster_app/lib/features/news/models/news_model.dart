class NewsModel {
  final String id;

  final String title;

  final String description;

  final String imageUrl;

  final DateTime publishedAt;

  NewsModel({
    required this.id,

    required this.title,

    required this.description,

    required this.imageUrl,

    required this.publishedAt,
  });

  factory NewsModel.fromJson(Map<String, dynamic> json) {
    return NewsModel(
      id: json['id'],

      title: json['title'],

      description: json['description'],

      imageUrl: json['imageUrl'],

      publishedAt: DateTime.parse(json['publishedAt']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,

      'title': title,

      'description': description,

      'imageUrl': imageUrl,

      'publishedAt': publishedAt.toIso8601String(),
    };
  }
}
