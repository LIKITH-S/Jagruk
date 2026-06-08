class SafetyModel {
  final String type;

  final String before;
  final String during;
  final String after;

  SafetyModel({
    required this.type,

    required this.before,
    required this.during,
    required this.after,
  });

  factory SafetyModel.fromJson(Map<String, dynamic> json) {
    return SafetyModel(
      type: json['type'],

      before: json['before'],
      during: json['during'],
      after: json['after'],
    );
  }
}
