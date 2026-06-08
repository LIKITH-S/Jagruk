import 'package:flutter_bloc/flutter_bloc.dart';

import '../services/safety_service.dart';

import 'safety_event.dart';
import 'safety_state.dart';

class SafetyBloc extends Bloc<SafetyEvent, SafetyState> {
  final SafetyService service;

  SafetyBloc(this.service) : super(SafetyLoading()) {
    on<LoadSafety>((event, emit) async {
      try {
        final instructions = await service.getSafetyInstructions();

        emit(SafetyLoaded(instructions));
      } catch (e) {
        emit(SafetyError("Failed to load safety data"));
      }
    });
  }
}
