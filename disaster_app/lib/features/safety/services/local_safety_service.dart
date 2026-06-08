import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/safety_model.dart';
import 'safety_service.dart';

class LocalSafetyService implements SafetyService {
  @override
  Future<List<SafetyModel>> getSafetyInstructions() async {
    final jsonString = await rootBundle.loadString(
      'assets/data/safety_data.json',
    );

    final List<dynamic> data = jsonDecode(jsonString);

    return data.map((e) => SafetyModel.fromJson(e)).toList();
  }
}
