import '../models/safety_model.dart';

abstract class SafetyService {
  Future<List<SafetyModel>> getSafetyInstructions();
}
