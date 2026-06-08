class ReportModel {
  final String id;

  final String description;

  final String disasterType;

  final double lat;
  final double lon;

  final DateTime timestamp;

  ReportModel({
    required this.id,

    required this.description,

    required this.disasterType,

    required this.lat,
    required this.lon,

    required this.timestamp,
  });

  factory ReportModel.fromJson(Map<String, dynamic> json) {
    return ReportModel(
      id: json['id'],

      description: json['description'],

      disasterType: json['disasterType'],

      lat: json['lat'],
      lon: json['lon'],

      timestamp: DateTime.parse(json['timestamp']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,

      'description': description,

      'disasterType': disasterType,

      'lat': lat,
      'lon': lon,

      'timestamp': timestamp.toIso8601String(),
    };
  }
}
