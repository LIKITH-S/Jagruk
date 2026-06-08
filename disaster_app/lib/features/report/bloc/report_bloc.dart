import 'package:flutter_bloc/flutter_bloc.dart';

import '../models/report_model.dart';

import '../services/report_service.dart';

import 'report_event.dart';
import 'report_state.dart';

class ReportBloc extends Bloc<ReportEvent, ReportState> {
  final ReportService service;

  ReportBloc(this.service) : super(ReportInitial()) {
    on<SubmitReport>((event, emit) async {
      emit(ReportLoading());

      try {
        final report = ReportModel(
          id: DateTime.now().millisecondsSinceEpoch.toString(),

          description: event.description,

          disasterType: event.disasterType,

          lat: event.lat,
          lon: event.lon,

          timestamp: DateTime.now(),
        );

        await service.submitReport(report);

        emit(ReportSuccess());
      } catch (e) {
        emit(ReportError("Failed to submit report"));
      }
    });
  }
}
