; PR6.6 Scheme frontend trace oracle. Run with the normal `meep -q` harness so
; wall-clock progress is not observable and static event gaps may be batched.

(set-param! dimensions 1)
(set-param! resolution 10)
(set! geometry-lattice (make lattice (size 0 0 2)))
(set! sources
      (list (make source
              (src (make gaussian-src (frequency 0.4) (fwidth 0.2)))
              (component Ex)
              (center 0 0 0))))

(define (assert condition message)
  (if (not condition) (error message)))

(define (run-trace force-one-step?)
  (if (not (null? fields)) (reset-meep))
  (init-fields)
  (let ((trace '()))
    (define (record name)
      (lambda (to-do)
        (set! trace (cons (list (meep-fields-t-get fields) to-do name) trace))))
    (let* ((scheduled
            (combine-step-funcs
             (at-beginning (record 'begin))
             (at-time 0.3 (record 'time))
             (at-every 0.2 (record 'every))
             (at-end (record 'end))))
           (funcs (if force-one-step?
                      (list scheduled (lambda () false))
                      (list scheduled))))
      (apply run-until (cons 1.0 funcs))
      (list (reverse trace)
            (meep-fields-t-get fields)
            (get-field-point Ex (vector3 0 0 0))))))

(let ((batched (run-trace false))
      (stepped (run-trace true)))
  (assert (equal? (car batched) (car stepped))
          "batched Scheme callback trace differs from one-step trace")
  (assert (= (cadr batched) (cadr stepped))
          "batched Scheme timestep differs from one-step trace")
  (assert (= (caddr batched) (caddr stepped))
          "batched Scheme field differs from one-step trace")
  (print "frontend-batching: PASS\n"))

; A rank-local event schedule would choose different native batch sizes if the
; frontend attempted batching under MPI. Completion at an identical timestep
; proves all ranks instead made the same sequence of collective step calls.
(if (> (meep-count-processors) 1)
    (begin
      (if (not (null? fields)) (reset-meep))
      (init-fields)
      (let ((hits 0)
            (local-event (+ 0.2 (* 0.1 (meep-my-rank)))))
        (run-until 1.0
                   (at-time local-event
                            (lambda () (set! hits (+ hits 1)))))
        (assert (= (meep-fields-t-get fields) 20)
                "MPI ranks completed different collective step counts")
        (assert (= hits 1) "rank-local Scheme callback did not fire once")
        (print "frontend-batching: MPI divergent schedule PASS\n"))))
